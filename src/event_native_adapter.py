from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from datasets.pedro_event_native import IMG_H, IMG_W, _build_stage_tensor, _build_target  # noqa: E402


@dataclass(frozen=True)
class EventNativeDetectorConfig:
    dts_ms: tuple[int, ...] = (10, 20, 40)
    anchor_dt_ms: int = 40
    image_size: int = 224
    num_time_bins: int = 4
    use_recency_channel: bool = True
    use_count_channel: bool = True
    resize_mode: str = "square"

    @property
    def event_channels(self) -> int:
        return (
            int(self.num_time_bins) * 2
            + 2 * int(bool(self.use_recency_channel))
            + int(bool(self.use_count_channel))
        )

    @property
    def event_plus_prior_channels(self) -> int:
        return self.event_channels + 4


def event_native_config_from_dict(cfg: dict) -> EventNativeDetectorConfig:
    dataset_cfg = cfg["dataset"]
    return EventNativeDetectorConfig(
        dts_ms=tuple(int(value) for value in dataset_cfg["dts_ms"]),
        anchor_dt_ms=int(dataset_cfg["anchor_dt_ms"]),
        image_size=int(dataset_cfg["image_size"]),
        num_time_bins=int(dataset_cfg["num_time_bins"]),
        use_recency_channel=bool(dataset_cfg["use_recency_channel"]),
        use_count_channel=bool(dataset_cfg["use_count_channel"]),
        resize_mode=str(dataset_cfg["resize_mode"]),
    )


class ManifestEventNativeSequenceDataset(Dataset):
    """Manifest-backed dataset that matches the existing event-native detector input.

    Returns:

    ```text
    stage_tensors: list[[11, 224, 224]]
    target: detector target dict with boxes scaled to 224x224
    meta: sequence/timestamp info for causal memory
    ```
    """

    def __init__(
        self,
        manifest_path: str | Path,
        detector_config: EventNativeDetectorConfig,
        *,
        split: str | Sequence[str] | None = None,
        subset: str | Sequence[str] | None = None,
        include_empty: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.detector_config = detector_config
        manifest = pd.read_csv(self.manifest_path)
        self.rows = self._filter_manifest(
            manifest,
            split=split,
            subset=subset,
            include_empty=include_empty,
        )
        self.rows = self.rows.sort_values(
            ["subset", "split", "sequence_index", "sequence_position", "frame_number"],
            kind="mergesort",
        ).reset_index(drop=True)
        self.sequence_to_indices = self._build_sequence_index()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        events = np.load(row["npy_path"], allow_pickle=False)
        t_end_us = int(row["window_end_us"])
        boxes_xyxy = self._boxes_from_json(row["boxes_json"])

        stage_tensors = [
            _build_stage_tensor(
                events=events,
                t_end_us=t_end_us,
                dt_us=int(dt_ms) * 1000,
                num_time_bins=int(self.detector_config.num_time_bins),
                use_recency_channel=bool(self.detector_config.use_recency_channel),
                use_count_channel=bool(self.detector_config.use_count_channel),
                image_size=int(self.detector_config.image_size),
                resize_mode=self.detector_config.resize_mode,
            )
            for dt_ms in self.detector_config.dts_ms
        ]
        target = _build_target(
            boxes_xyxy=boxes_xyxy,
            image_id=index + 1,
            image_size=int(self.detector_config.image_size),
            resize_mode=self.detector_config.resize_mode,
        )
        target["key"] = f"{row['subset']}:{row['split']}:{row['frame_stem']}"

        meta = {
            "index": index,
            "subset": row["subset"],
            "split": row["split"],
            "sequence_id": row["sequence_id"],
            "sequence_index": int(row["sequence_index"]),
            "sequence_position": int(row["sequence_position"]),
            "is_sequence_start": bool(row["is_sequence_start"]),
            "frame_stem": row["frame_stem"],
            "frame_number": int(row["frame_number"]),
            "window_start_us": int(row["window_start_us"]),
            "window_end_us": int(row["window_end_us"]),
            "window_start_time": int(row["window_start_us"]) / 1_000_000.0,
            "window_end_time": int(row["window_end_us"]) / 1_000_000.0,
            "delta_t_s": None if pd.isna(row["delta_t_s"]) else float(row["delta_t_s"]),
            "duration_us": int(row["duration_us"]),
            "num_events": int(row["num_events"]),
            "num_boxes": int(row["num_boxes"]),
            "npy_path": row["npy_path"],
            "xml_path": row["xml_path"],
        }
        return stage_tensors, target, meta

    @property
    def sequence_ids(self) -> list[str]:
        return list(self.sequence_to_indices)

    def iter_sequence_indices(self) -> Iterator[list[int]]:
        for indices in self.sequence_to_indices.values():
            yield list(indices)

    def _build_sequence_index(self) -> dict[str, list[int]]:
        sequence_to_indices: dict[str, list[int]] = {}
        for index, row in self.rows.iterrows():
            sequence_to_indices.setdefault(row["sequence_id"], []).append(index)

        for sequence_id, indices in sequence_to_indices.items():
            positions = self.rows.iloc[indices]["sequence_position"].tolist()
            expected = list(range(len(indices)))
            if positions != expected:
                raise ValueError(f"non-contiguous positions in sequence {sequence_id}")
        return sequence_to_indices

    @staticmethod
    def _boxes_from_json(boxes_json: str) -> list[tuple[float, float, float, float]]:
        boxes = []
        for obj in json.loads(boxes_json):
            if str(obj.get("class", "")).lower() != "person":
                continue
            x1 = max(0.0, min(float(obj["xmin"]), float(IMG_W)))
            y1 = max(0.0, min(float(obj["ymin"]), float(IMG_H)))
            x2 = max(0.0, min(float(obj["xmax"]), float(IMG_W)))
            y2 = max(0.0, min(float(obj["ymax"]), float(IMG_H)))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    @staticmethod
    def _filter_manifest(
        manifest: pd.DataFrame,
        *,
        split: str | Sequence[str] | None,
        subset: str | Sequence[str] | None,
        include_empty: bool,
    ) -> pd.DataFrame:
        filtered = manifest.copy()
        if split is not None:
            splits = {split} if isinstance(split, str) else set(split)
            filtered = filtered[filtered["split"].isin(splits)]
        if subset is not None:
            subsets = {subset} if isinstance(subset, str) else set(subset)
            filtered = filtered[filtered["subset"].isin(subsets)]
        if not include_empty:
            filtered = filtered[filtered["num_boxes"] > 0]
        return filtered


class SequentialEventNativeBatchSampler(Sampler[list[int]]):
    """Yield contiguous event-native batches without crossing sequence boundaries."""

    def __init__(
        self,
        dataset: ManifestEventNativeSequenceDataset,
        *,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle_sequences: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle_sequences = bool(shuffle_sequences)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        sequences = list(self.dataset.iter_sequence_indices())
        if self.shuffle_sequences:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(sequences)

        for sequence_indices in sequences:
            for start in range(0, len(sequence_indices), self.batch_size):
                batch = sequence_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for sequence_indices in self.dataset.iter_sequence_indices():
            full, remainder = divmod(len(sequence_indices), self.batch_size)
            total += full
            if remainder and not self.drop_last:
                total += 1
        return total


def collate_event_native_memory(batch):
    stage_sequences, targets, metas = zip(*batch)
    num_stages = len(stage_sequences[0])
    collated_stages = []
    for stage_idx in range(num_stages):
        collated_stages.append(torch.stack([sequence[stage_idx] for sequence in stage_sequences], dim=0))
    return collated_stages, list(targets), list(metas)


def resize_prior_maps_to_detector(prior_maps: torch.Tensor, image_size: int) -> torch.Tensor:
    if prior_maps.ndim != 4:
        raise ValueError(f"expected prior maps [B,4,H,W], got {tuple(prior_maps.shape)}")
    return torch.nn.functional.interpolate(
        prior_maps,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    )


def append_priors_to_stage_tensors(
    stage_tensors: Sequence[torch.Tensor],
    prior_maps: torch.Tensor | Sequence[torch.Tensor],
    *,
    image_size: int,
) -> list[torch.Tensor]:
    if isinstance(prior_maps, torch.Tensor):
        detector_priors = [resize_prior_maps_to_detector(prior_maps, image_size=image_size)] * len(stage_tensors)
    else:
        if len(prior_maps) != len(stage_tensors):
            raise ValueError(
                f"expected {len(stage_tensors)} prior tensors, got {len(prior_maps)}"
            )
        detector_priors = [
            resize_prior_maps_to_detector(prior, image_size=image_size)
            for prior in prior_maps
        ]
    return [
        torch.cat([stage, prior.to(device=stage.device, dtype=stage.dtype)], dim=1)
        for stage, prior in zip(stage_tensors, detector_priors)
    ]


def load_checkpoint_with_expanded_input(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, int]:
    """Load an event-only checkpoint into an event-plus-prior model.

    Compatible weights are copied directly. A convolution whose checkpoint input
    channels are fewer than the model input channels is expanded by copying the
    old channels and zero-initializing the new prior channels.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()

    compatible = {}
    expanded = 0
    skipped = 0
    for key, value in state_dict.items():
        if key not in model_state:
            skipped += 1
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            compatible[key] = value
            continue
        can_expand_conv_input = (
            value.ndim == 4
            and target.ndim == 4
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
            and value.shape[1] < target.shape[1]
        )
        if can_expand_conv_input:
            expanded_weight = target.clone()
            expanded_weight.zero_()
            expanded_weight[:, : value.shape[1], :, :] = value
            compatible[key] = expanded_weight
            expanded += 1
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "expanded_input_convs": expanded,
        "skipped": skipped,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
