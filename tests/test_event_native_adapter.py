import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.event_memory import EventMemoryConfig, UncertaintyAwareEventMemory
from src.event_native_adapter import (
    EventNativeDetectorConfig,
    ManifestEventNativeSequenceDataset,
    SequentialEventNativeBatchSampler,
    append_priors_to_stage_tensors,
    collate_event_native_memory,
)


def test_event_native_adapter_shapes_and_sequence_batching(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    cfg = EventNativeDetectorConfig(
        dts_ms=(10, 20, 40),
        anchor_dt_ms=40,
        image_size=32,
        num_time_bins=2,
        use_recency_channel=True,
        use_count_channel=True,
    )
    dataset = ManifestEventNativeSequenceDataset(manifest_path, cfg, split="train", subset="main")

    assert cfg.event_channels == 7
    assert cfg.event_plus_prior_channels == 11
    assert len(dataset) == 2
    assert dataset.sequence_ids == ["main_train_seq0000"]

    stages, target, meta = dataset[0]
    assert [tuple(stage.shape) for stage in stages] == [(7, 32, 32), (7, 32, 32), (7, 32, 32)]
    assert target["boxes"].shape == (1, 4)
    assert meta["is_sequence_start"] is True

    loader = DataLoader(
        dataset,
        batch_sampler=SequentialEventNativeBatchSampler(dataset, batch_size=1),
        collate_fn=collate_event_native_memory,
    )
    batch_stages, targets, metas = next(iter(loader))
    memory = UncertaintyAwareEventMemory(EventMemoryConfig(height=32, width=32), batch_size=1)
    priors = memory.make_priors(batch_stages[-1], window_end_time=metas[0]["window_end_time"])
    augmented = append_priors_to_stage_tensors(batch_stages, priors, image_size=32)

    assert priors.shape == (1, 4, 32, 32)
    assert [tuple(stage.shape) for stage in augmented] == [(1, 11, 32, 32)] * 3
    assert targets[0]["key"] == "main:train:frame0000000"


def _write_manifest(tmp_path: Path) -> Path:
    rows = []
    for idx in range(2):
        frame_stem = f"frame{idx:07d}"
        start_us = idx * 50_000
        end_us = start_us + 40_000
        events = np.array(
            [
                [start_us, 10, 20, 1],
                [start_us + 10_000, 12, 22, 0],
                [end_us, 15, 25, 1],
            ],
            dtype=np.int64,
        )
        npy_path = tmp_path / f"{frame_stem}.npy"
        xml_path = tmp_path / f"{frame_stem}.xml"
        np.save(npy_path, events)
        xml_path.write_text("<annotation />")
        boxes = [
            {
                "class": "person",
                "xmin": 10,
                "ymin": 20,
                "xmax": 40,
                "ymax": 80,
                "difficult": 0,
                "truncated": 0,
            }
        ]
        rows.append(
            {
                "subset": "main",
                "split": "train",
                "sequence_id": "main_train_seq0000",
                "sequence_index": 0,
                "sequence_position": idx,
                "is_sequence_start": int(idx == 0),
                "sequence_break_reason": "first_accepted_window" if idx == 0 else "",
                "frame_stem": frame_stem,
                "frame_number": idx,
                "window_start_us": start_us,
                "window_end_us": end_us,
                "duration_us": 40_000,
                "delta_t_s": "" if idx == 0 else 0.05,
                "num_events": len(events),
                "num_boxes": len(boxes),
                "boxes_json": json.dumps(boxes),
                "npy_path": str(npy_path),
                "xml_path": str(xml_path),
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
