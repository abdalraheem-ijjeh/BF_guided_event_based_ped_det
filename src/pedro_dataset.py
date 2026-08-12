from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class EventRepresentationConfig:
    height: int = 260
    width: int = 346
    representation: str = "polarity_count"
    num_bins: int = 5
    normalize: str = "log1p"
    dtype: torch.dtype = torch.float32

    @property
    def channels(self) -> int:
        if self.representation == "polarity_count":
            return 2
        if self.representation == "voxel_grid":
            return 2 * self.num_bins
        raise ValueError(f"unsupported representation: {self.representation}")


def build_event_representation(
    events: np.ndarray,
    config: EventRepresentationConfig | None = None,
) -> torch.Tensor:
    """Convert PEDRo `[timestamp, x, y, polarity]` events into `[C, H, W]`.

    Supported representations:

    - `polarity_count`: two channels, one for polarity 0 and one for polarity 1.
    - `voxel_grid`: `2 * num_bins` channels, with temporal bins per polarity.
    """
    cfg = config or EventRepresentationConfig()
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"events must have shape [N, 4], got {events.shape}")
    if events.shape[0] == 0:
        raise ValueError("events must not be empty")

    tensor = torch.zeros(
        (cfg.channels, cfg.height, cfg.width),
        dtype=cfg.dtype,
    )

    timestamps = torch.as_tensor(events[:, 0].astype(np.int64), dtype=torch.float64)
    xs = torch.as_tensor(events[:, 1].astype(np.int64), dtype=torch.long)
    ys = torch.as_tensor(events[:, 2].astype(np.int64), dtype=torch.long)
    polarities = torch.as_tensor(events[:, 3].astype(np.int64), dtype=torch.long)

    valid = (
        (xs >= 0)
        & (xs < cfg.width)
        & (ys >= 0)
        & (ys < cfg.height)
        & ((polarities == 0) | (polarities == 1))
    )
    if not bool(valid.all()):
        xs = xs[valid]
        ys = ys[valid]
        polarities = polarities[valid]
        timestamps = timestamps[valid]

    if xs.numel() == 0:
        return tensor

    if cfg.representation == "polarity_count":
        channels = polarities
    elif cfg.representation == "voxel_grid":
        t0 = timestamps[0]
        t1 = timestamps[-1]
        if float(t1 - t0) <= 0:
            bins = torch.zeros_like(polarities)
        else:
            normalized = (timestamps - t0) / (t1 - t0)
            bins = torch.clamp((normalized * cfg.num_bins).long(), 0, cfg.num_bins - 1)
        channels = polarities * cfg.num_bins + bins
    else:
        raise ValueError(f"unsupported representation: {cfg.representation}")

    flat_indices = channels * (cfg.height * cfg.width) + ys * cfg.width + xs
    flat = tensor.view(-1)
    flat.index_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=cfg.dtype))
    return normalize_event_tensor(tensor, cfg.normalize)


def normalize_event_tensor(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return tensor
    if mode == "log1p":
        return torch.log1p(tensor)
    if mode == "max":
        max_value = tensor.amax()
        return tensor / max_value.clamp_min(1.0)
    raise ValueError(f"unsupported normalization: {mode}")


class PedroManifestDataset(Dataset):
    """Manifest-backed PEDRo loader for causal event-memory training."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | Sequence[str] | None = None,
        subset: str | Sequence[str] | None = None,
        representation_config: EventRepresentationConfig | None = None,
        class_to_label: dict[str, int] | None = None,
        load_events: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.representation_config = representation_config or EventRepresentationConfig()
        self.class_to_label = class_to_label or {"person": 1}
        self.load_events = load_events

        manifest = pd.read_csv(self.manifest_path)
        self.rows = self._filter_manifest(manifest, split=split, subset=subset)
        self.rows = self.rows.sort_values(
            ["subset", "split", "sequence_index", "sequence_position", "frame_number"],
            kind="mergesort",
        ).reset_index(drop=True)
        self.sequence_to_indices = self._build_sequence_index()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        boxes, labels, difficult, truncated = self._parse_targets(row["boxes_json"])

        sample: dict[str, object] = {
            "index": index,
            "subset": row["subset"],
            "split": row["split"],
            "sequence_id": row["sequence_id"],
            "sequence_index": int(row["sequence_index"]),
            "sequence_position": int(row["sequence_position"]),
            "is_sequence_start": bool(row["is_sequence_start"]),
            "frame_stem": row["frame_stem"],
            "frame_number": int(row["frame_number"]),
            "npy_path": row["npy_path"],
            "xml_path": row["xml_path"],
            "window_start_us": int(row["window_start_us"]),
            "window_end_us": int(row["window_end_us"]),
            "window_start_time": int(row["window_start_us"]) / 1_000_000.0,
            "window_end_time": int(row["window_end_us"]) / 1_000_000.0,
            "duration_us": int(row["duration_us"]),
            "delta_t_s": self._optional_float(row["delta_t_s"]),
            "num_events": int(row["num_events"]),
            "boxes": boxes,
            "labels": labels,
            "difficult": difficult,
            "truncated": truncated,
            "num_boxes": int(row["num_boxes"]),
        }

        if self.load_events:
            events = np.load(row["npy_path"], allow_pickle=False)
            sample["raw_events"] = events
            sample["event_tensor"] = build_event_representation(
                events,
                self.representation_config,
            )

        return sample

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

    def _parse_targets(
        self,
        boxes_json: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        objects = json.loads(boxes_json)
        boxes: list[list[float]] = []
        labels: list[int] = []
        difficult: list[int] = []
        truncated: list[int] = []

        for obj in objects:
            class_name = obj["class"]
            if class_name not in self.class_to_label:
                continue
            boxes.append(
                [
                    float(obj["xmin"]),
                    float(obj["ymin"]),
                    float(obj["xmax"]),
                    float(obj["ymax"]),
                ]
            )
            labels.append(self.class_to_label[class_name])
            difficult.append(int(obj.get("difficult", 0)))
            truncated.append(int(obj.get("truncated", 0)))

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.long),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(difficult, dtype=torch.long),
            torch.tensor(truncated, dtype=torch.long),
        )

    @staticmethod
    def _filter_manifest(
        manifest: pd.DataFrame,
        *,
        split: str | Sequence[str] | None,
        subset: str | Sequence[str] | None,
    ) -> pd.DataFrame:
        filtered = manifest.copy()
        if split is not None:
            splits = {split} if isinstance(split, str) else set(split)
            filtered = filtered[filtered["split"].isin(splits)]
        if subset is not None:
            subsets = {subset} if isinstance(subset, str) else set(subset)
            filtered = filtered[filtered["subset"].isin(subsets)]
        return filtered

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if pd.isna(value):
            return None
        return float(value)


class SequentialSequenceBatchSampler(Sampler[list[int]]):
    """Yield contiguous batches without crossing inferred sequence boundaries."""

    def __init__(
        self,
        dataset: PedroManifestDataset,
        *,
        batch_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        for sequence_indices in self.dataset.iter_sequence_indices():
            for start in range(0, len(sequence_indices), self.batch_size):
                batch = sequence_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for sequence_indices in self.dataset.iter_sequence_indices():
            full_batches, remainder = divmod(len(sequence_indices), self.batch_size)
            total += full_batches
            if remainder and not self.drop_last:
                total += 1
        return total


def pedro_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate PEDRo samples while preserving variable numbers of boxes."""
    if not batch:
        raise ValueError("batch must not be empty")

    collated: dict[str, object] = {}
    passthrough_keys = {
        "index",
        "subset",
        "split",
        "sequence_id",
        "sequence_index",
        "sequence_position",
        "is_sequence_start",
        "frame_stem",
        "frame_number",
        "npy_path",
        "xml_path",
        "window_start_us",
        "window_end_us",
        "window_start_time",
        "window_end_time",
        "duration_us",
        "delta_t_s",
        "num_events",
        "num_boxes",
    }

    for key in passthrough_keys:
        collated[key] = [sample[key] for sample in batch]

    collated["boxes"] = [sample["boxes"] for sample in batch]
    collated["labels"] = [sample["labels"] for sample in batch]
    collated["difficult"] = [sample["difficult"] for sample in batch]
    collated["truncated"] = [sample["truncated"] for sample in batch]

    if "event_tensor" in batch[0]:
        collated["event_tensor"] = torch.stack(
            [sample["event_tensor"] for sample in batch], dim=0
        )
    if "raw_events" in batch[0]:
        collated["raw_events"] = [sample["raw_events"] for sample in batch]

    return collated
