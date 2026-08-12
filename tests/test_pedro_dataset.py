import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.pedro_dataset import (
    EventRepresentationConfig,
    PedroManifestDataset,
    SequentialSequenceBatchSampler,
    build_event_representation,
    pedro_collate,
)


def test_build_polarity_count_representation() -> None:
    events = np.array(
        [
            [100, 1, 2, 0],
            [101, 1, 2, 1],
            [102, 1, 2, 1],
            [103, 3, 4, 0],
        ],
        dtype=np.int64,
    )
    cfg = EventRepresentationConfig(height=6, width=7, representation="polarity_count", normalize="none")

    tensor = build_event_representation(events, cfg)

    assert tensor.shape == (2, 6, 7)
    assert tensor[0, 2, 1].item() == 1
    assert tensor[1, 2, 1].item() == 2
    assert tensor[0, 4, 3].item() == 1


def test_build_voxel_grid_representation() -> None:
    events = np.array(
        [
            [0, 1, 1, 0],
            [5, 2, 1, 0],
            [10, 3, 1, 1],
        ],
        dtype=np.int64,
    )
    cfg = EventRepresentationConfig(
        height=4,
        width=5,
        representation="voxel_grid",
        num_bins=2,
        normalize="none",
    )

    tensor = build_event_representation(events, cfg)

    assert tensor.shape == (4, 4, 5)
    assert tensor[0, 1, 1].item() == 1
    assert tensor[1, 1, 2].item() == 1
    assert tensor[3, 1, 3].item() == 1


def test_manifest_dataset_and_sampler_preserve_sequences(tmp_path: Path) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    cfg = EventRepresentationConfig(height=6, width=7, representation="polarity_count", normalize="none")
    dataset = PedroManifestDataset(manifest_path, split="train", representation_config=cfg)

    assert len(dataset) == 4
    assert dataset.sequence_ids == ["main_train_seq0000", "main_train_seq0001"]

    first = dataset[0]
    assert first["is_sequence_start"] is True
    assert first["sequence_position"] == 0
    assert first["window_end_time"] == 0.04
    assert first["event_tensor"].shape == (2, 6, 7)
    assert torch.equal(first["boxes"], torch.tensor([[1.0, 1.0, 4.0, 5.0]]))
    assert torch.equal(first["labels"], torch.tensor([1]))

    empty = dataset[3]
    assert empty["num_boxes"] == 0
    assert empty["boxes"].shape == (0, 4)
    assert empty["labels"].shape == (0,)

    sampler = SequentialSequenceBatchSampler(dataset, batch_size=2)
    assert list(iter(sampler)) == [[0, 1], [2], [3]]

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pedro_collate)
    batches = list(loader)
    assert batches[0]["event_tensor"].shape == (2, 2, 6, 7)
    assert batches[0]["sequence_id"] == ["main_train_seq0000", "main_train_seq0000"]
    assert batches[1]["is_sequence_start"] == [False]
    assert batches[2]["is_sequence_start"] == [True]


def _write_synthetic_manifest(tmp_path: Path) -> Path:
    rows = []
    event_specs = [
        ("frame0000000", "main_train_seq0000", 0, 0, 1, None, None, [[1, 1, 4, 5]]),
        ("frame0000001", "main_train_seq0000", 0, 1, 0, 10_000, 0.01, [[2, 1, 5, 5]]),
        ("frame0000002", "main_train_seq0000", 0, 2, 0, 10_000, 0.01, [[2, 2, 5, 5]]),
        ("frame0000010", "main_train_seq0001", 1, 0, 1, 2_000_000, 2.0, []),
    ]

    for frame_stem, sequence_id, sequence_index, sequence_position, is_start, gap_us, delta_t, boxes in event_specs:
        frame_number = int(frame_stem.replace("frame", ""))
        start_us = frame_number * 100_000
        end_us = start_us + 40_000
        npy_path = tmp_path / f"{frame_stem}.npy"
        xml_path = tmp_path / f"{frame_stem}.xml"
        events = np.array(
            [
                [start_us, 1, 2, 0],
                [start_us + 1, 1, 2, 1],
                [end_us, 2, 3, 1],
            ],
            dtype=np.int64,
        )
        np.save(npy_path, events)
        xml_path.write_text("<annotation />")

        objects = [
            {
                "class": "person",
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3],
                "difficult": 0,
                "truncated": 0,
            }
            for box in boxes
        ]
        rows.append(
            {
                "subset": "main",
                "split": "train",
                "sequence_id": sequence_id,
                "sequence_index": sequence_index,
                "sequence_position": sequence_position,
                "is_sequence_start": is_start,
                "sequence_break_reason": "first_accepted_window" if is_start else "",
                "frame_stem": frame_stem,
                "frame_number": frame_number,
                "previous_accepted_frame_stem": "",
                "frame_number_gap_from_previous": "",
                "has_missing_or_rejected_between": "",
                "gap_from_previous_us": "" if gap_us is None else gap_us,
                "delta_t_s": "" if delta_t is None else delta_t,
                "window_start_us": start_us,
                "window_end_us": end_us,
                "duration_us": 40_000,
                "num_events": len(events),
                "event_timestamps_monotonic": True,
                "width": 7,
                "height": 6,
                "depth": 1,
                "num_boxes": len(objects),
                "class_counts_json": json.dumps({"person": len(objects)} if objects else {}),
                "boxes_json": json.dumps(objects),
                "has_invalid_boxes": False,
                "invalid_boxes_json": "[]",
                "npy_path": str(npy_path),
                "xml_path": str(xml_path),
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
