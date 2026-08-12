"""PEDRo event-native dataset entry point.

This loader consumes raw PEDRo event files and XML annotations directly:

  PEDRo_original/
    numpy/
      train|val|test/
        frame0000000.npy
    xml/
      train|val|test/
        frame0000000.xml

For each anchor sample, the loader builds end-aligned stage tensors over
`[t-dt, t]` and returns a list of multi-channel event-native tensors plus a
COCO-style target dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset


IMG_W = 346
IMG_H = 260
RAW_SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


@dataclass(frozen=True)
class PedroEventNativeConfig:
    dataset_root: str
    split: str
    dts_ms: Sequence[int]
    anchor_dt_ms: int
    image_size: int
    num_time_bins: int = 4
    use_recency_channel: bool = True
    use_count_channel: bool = True
    resize_mode: str = "square"
    person_classes: Tuple[str, ...] = ("person", "pedestrian", "rider")
    min_box_area: float = 1.0
    require_complete_group: bool = True
    include_empty_frames: bool = False


def _resolve_split_name(split: str) -> str:
    key = str(split).strip().lower()
    if key not in RAW_SPLIT_MAP:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {sorted(RAW_SPLIT_MAP)}")
    return RAW_SPLIT_MAP[key]


def _list_frame_stems(np_dir: Path, xml_dir: Path) -> List[str]:
    np_stems = {p.stem for p in np_dir.glob("frame*.npy")}
    xml_stems = {p.stem for p in xml_dir.glob("frame*.xml")}
    return sorted(np_stems & xml_stems)


def _load_events(events_path: Path) -> np.ndarray:
    try:
        events = np.load(events_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Failed to load events from {events_path}: {exc}") from exc
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"{events_path} has unexpected shape {events.shape}; expected (N,4)")
    return events


def _parse_xml_objects(xml_path: Path, person_classes: set[str], min_box_area: float) -> List[Tuple[float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    boxes: List[Tuple[float, float, float, float]] = []
    for obj in root.findall("object"):
        class_name = str(obj.findtext("name", default="")).strip().lower()
        if class_name not in person_classes:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", default="0"))
        ymin = float(bnd.findtext("ymin", default="0"))
        xmax = float(bnd.findtext("xmax", default="0"))
        ymax = float(bnd.findtext("ymax", default="0"))
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        if (w * h) < float(min_box_area):
            continue
        x1 = max(0.0, min(xmin, float(IMG_W)))
        y1 = max(0.0, min(ymin, float(IMG_H)))
        x2 = max(0.0, min(xmax, float(IMG_W)))
        y2 = max(0.0, min(ymax, float(IMG_H)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def _resize_tensor_square(tensor: torch.Tensor, image_size: int) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
    return torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _build_target(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    image_id: int,
    image_size: int,
    resize_mode: str,
) -> Dict[str, torch.Tensor | str]:
    if resize_mode != "square":
        raise NotImplementedError(f"Unsupported resize_mode '{resize_mode}'")

    scale_x = float(image_size) / float(IMG_W)
    scale_y = float(image_size) / float(IMG_H)

    boxes_scaled: List[List[float]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        sx1 = max(0.0, min(x1 * scale_x, float(image_size)))
        sy1 = max(0.0, min(y1 * scale_y, float(image_size)))
        sx2 = max(0.0, min(x2 * scale_x, float(image_size)))
        sy2 = max(0.0, min(y2 * scale_y, float(image_size)))
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        boxes_scaled.append([sx1, sy1, sx2, sy2])

    if boxes_scaled:
        boxes_tensor = torch.tensor(boxes_scaled, dtype=torch.float32)
        labels_tensor = torch.zeros((len(boxes_scaled),), dtype=torch.int64)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)

    return {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
        "orig_size": torch.tensor([IMG_H, IMG_W], dtype=torch.int64),
        "size": torch.tensor([int(image_size), int(image_size)], dtype=torch.int64),
    }


def _build_stage_tensor(
    events: np.ndarray,
    t_end_us: int,
    dt_us: int,
    num_time_bins: int,
    use_recency_channel: bool,
    use_count_channel: bool,
    image_size: int,
    resize_mode: str,
) -> torch.Tensor:
    if resize_mode != "square":
        raise NotImplementedError(f"Unsupported resize_mode '{resize_mode}'")

    t_start_us = int(t_end_us) - int(dt_us)
    t = events[:, 0].astype(np.int64, copy=False)
    x = events[:, 1].astype(np.int64, copy=False)
    y = events[:, 2].astype(np.int64, copy=False)
    p = events[:, 3].astype(np.int64, copy=False)

    valid = (
        (t >= t_start_us)
        & (t <= t_end_us)
        & (x >= 0)
        & (x < IMG_W)
        & (y >= 0)
        & (y < IMG_H)
    )

    x = x[valid]
    y = y[valid]
    t = t[valid]
    p = p[valid]

    num_base_channels = int(num_time_bins) * 2
    extra_channels = (2 * int(bool(use_recency_channel))) + int(bool(use_count_channel))
    channels = np.zeros((num_base_channels + extra_channels, IMG_H, IMG_W), dtype=np.float32)

    if x.size > 0:
        bin_width = max(1.0, float(dt_us) / float(num_time_bins))
        rel_t = np.clip((t - int(t_start_us)).astype(np.float32), 0.0, float(dt_us) - 1e-6)
        bin_idx = np.minimum((rel_t / bin_width).astype(np.int64), int(num_time_bins) - 1)

        pos_mask = p > 0
        neg_mask = ~pos_mask

        for pol_mask, channel_offset in ((pos_mask, 0), (neg_mask, int(num_time_bins))):
            if not np.any(pol_mask):
                continue
            flat = y[pol_mask] * IMG_W + x[pol_mask]
            bins = bin_idx[pol_mask]
            for b in range(int(num_time_bins)):
                sel = bins == b
                if not np.any(sel):
                    continue
                counts = np.zeros((IMG_H * IMG_W,), dtype=np.float32)
                np.add.at(counts, flat[sel], 1.0)
                channels[channel_offset + b] = counts.reshape(IMG_H, IMG_W)

        voxel_max = float(channels[:num_base_channels].max())
        if voxel_max > 0.0:
            channels[:num_base_channels] = np.log1p(channels[:num_base_channels]) / np.log1p(voxel_max)

        next_channel = num_base_channels
        if use_recency_channel:
            flat = y * IMG_W + x
            for pol_mask in (pos_mask, neg_mask):
                latest = np.full((IMG_H * IMG_W,), -1, dtype=np.int64)
                if np.any(pol_mask):
                    np.maximum.at(latest, flat[pol_mask], t[pol_mask])
                latest = latest.reshape(IMG_H, IMG_W)
                recency = np.zeros((IMG_H, IMG_W), dtype=np.float32)
                seen = latest >= 0
                if np.any(seen):
                    recency[seen] = (latest[seen] - int(t_start_us)).astype(np.float32) / float(dt_us)
                channels[next_channel] = np.clip(recency, 0.0, 1.0)
                next_channel += 1

        if use_count_channel:
            counts = np.zeros((IMG_H * IMG_W,), dtype=np.float32)
            np.add.at(counts, y * IMG_W + x, 1.0)
            counts = counts.reshape(IMG_H, IMG_W)
            if float(counts.max()) > 0.0:
                counts = np.log1p(counts) / np.log1p(float(counts.max()))
            channels[next_channel] = counts

    tensor = torch.from_numpy(channels)
    return _resize_tensor_square(tensor, image_size=image_size)


class PedroEventNativeDataset(Dataset):
    def __init__(self, cfg: PedroEventNativeConfig) -> None:
        self.cfg = cfg
        self.dataset_root = Path(cfg.dataset_root)
        self.split = _resolve_split_name(cfg.split)
        self.dts_ms = [int(dt) for dt in cfg.dts_ms]
        self.anchor_dt_ms = int(cfg.anchor_dt_ms)
        if self.anchor_dt_ms not in self.dts_ms:
            raise ValueError(f"anchor_dt_ms={self.anchor_dt_ms} must be contained in dts_ms={self.dts_ms}")
        self.image_size = int(cfg.image_size)
        self.person_classes = {c.strip().lower() for c in cfg.person_classes}

        np_dir = self.dataset_root / "numpy" / self.split
        xml_dir = self.dataset_root / "xml" / self.split
        if not np_dir.exists():
            raise FileNotFoundError(f"Missing PEDRo numpy split: {np_dir}")
        if not xml_dir.exists():
            raise FileNotFoundError(f"Missing PEDRo xml split: {xml_dir}")

        frame_stems = _list_frame_stems(np_dir=np_dir, xml_dir=xml_dir)
        self.samples: List[Dict[str, object]] = []
        self.skipped_event_files: List[Tuple[str, str]] = []

        for image_id, frame_stem in enumerate(frame_stems, start=1):
            events_path = np_dir / f"{frame_stem}.npy"
            xml_path = xml_dir / f"{frame_stem}.xml"
            try:
                _ = _load_events(events_path)
            except Exception as exc:
                self.skipped_event_files.append((frame_stem, str(exc)))
                continue
            boxes_xyxy = _parse_xml_objects(
                xml_path=xml_path,
                person_classes=self.person_classes,
                min_box_area=float(cfg.min_box_area),
            )
            if boxes_xyxy or bool(cfg.include_empty_frames):
                self.samples.append(
                    {
                        "frame_stem": frame_stem,
                        "events_path": events_path,
                        "xml_path": xml_path,
                        "boxes_xyxy": boxes_xyxy,
                        "image_id": image_id,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def summary(self) -> Dict[str, int]:
        empty = sum(1 for sample in self.samples if not sample["boxes_xyxy"])
        return {
            "samples": len(self.samples),
            "empty_samples": empty,
            "non_empty_samples": len(self.samples) - empty,
            "skipped_event_files": len(self.skipped_event_files),
        }

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        events = _load_events(Path(sample["events_path"]))
        if len(events) == 0:
            raise ValueError(f"Empty event file: {sample['events_path']}")

        t_end_us = int(events[-1, 0])
        stage_tensors: List[torch.Tensor] = []
        for dt_ms in self.dts_ms:
            stage_tensors.append(
                _build_stage_tensor(
                    events=events,
                    t_end_us=t_end_us,
                    dt_us=int(dt_ms * 1000),
                    num_time_bins=int(self.cfg.num_time_bins),
                    use_recency_channel=bool(self.cfg.use_recency_channel),
                    use_count_channel=bool(self.cfg.use_count_channel),
                    image_size=self.image_size,
                    resize_mode=self.cfg.resize_mode,
                )
            )

        target = _build_target(
            boxes_xyxy=list(sample["boxes_xyxy"]),
            image_id=int(sample["image_id"]),
            image_size=self.image_size,
            resize_mode=self.cfg.resize_mode,
        )
        target["key"] = str(sample["frame_stem"])
        return stage_tensors, target


def collate_pedro_event_native(batch):
    stage_sequences, targets = zip(*batch)
    num_stages = len(stage_sequences[0])
    collated_stages = []
    for stage_idx in range(num_stages):
        collated_stages.append(torch.stack([sequence[stage_idx] for sequence in stage_sequences], dim=0))
    return collated_stages, list(targets)
