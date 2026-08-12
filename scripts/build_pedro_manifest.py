from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FRAME_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class DatasetPart:
    subset: str
    split: str
    numpy_dir: Path
    xml_dir: Path


def dataset_parts(root: Path) -> list[DatasetPart]:
    return [
        DatasetPart("main", "train", root / "numpy/train", root / "xml/train"),
        DatasetPart("main", "val", root / "numpy/val", root / "xml/val"),
        DatasetPart("main", "test", root / "numpy/test", root / "xml/test"),
        DatasetPart(
            "live_person_only",
            "train",
            root / "live_annotated_dataset_person_only/numpy/train",
            root / "live_annotated_dataset_person_only/xml/train",
        ),
        DatasetPart(
            "no_person_train",
            "train",
            root / "NO_PERSPNS/PEDRo_original/with_no_persons/numpy/train",
            root / "NO_PERSPNS/PEDRo_original/with_no_persons/xml/train",
        ),
        DatasetPart(
            "no_person_val",
            "val",
            root / "valid_set_no_person/numpy/val",
            root / "valid_set_no_person/xml/val",
        ),
    ]


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.stem)
    if match is None:
        raise ValueError(f"could not parse numeric frame id from {path.name}")
    return int(match.group(1))


def parse_voc_xml(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("missing <size>")

    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    depth = int(float(size.findtext("depth", "0")))
    boxes: list[dict[str, Any]] = []

    for obj in root.findall("object"):
        name = obj.findtext("name", default="")
        difficult = int(float(obj.findtext("difficult", default="0")))
        truncated = int(float(obj.findtext("truncated", default="0")))
        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError("object missing <bndbox>")
        x_min = int(float(bbox.findtext("xmin", "0")))
        y_min = int(float(bbox.findtext("ymin", "0")))
        x_max = int(float(bbox.findtext("xmax", "0")))
        y_max = int(float(bbox.findtext("ymax", "0")))
        boxes.append(
            {
                "class": name,
                "xmin": x_min,
                "ymin": y_min,
                "xmax": x_max,
                "ymax": y_max,
                "difficult": difficult,
                "truncated": truncated,
            }
        )

    invalid_boxes = []
    for box in boxes:
        if not (
            0 <= box["xmin"] <= box["xmax"] <= width
            and 0 <= box["ymin"] <= box["ymax"] <= height
        ):
            invalid_boxes.append(box)

    class_counts = Counter(box["class"] for box in boxes)

    return {
        "width": width,
        "height": height,
        "depth": depth,
        "num_boxes": len(boxes),
        "class_counts_json": json.dumps(dict(sorted(class_counts.items())), separators=(",", ":")),
        "boxes_json": json.dumps(boxes, separators=(",", ":")),
        "has_invalid_boxes": bool(invalid_boxes),
        "invalid_boxes_json": json.dumps(invalid_boxes, separators=(",", ":")),
    }


def load_event_stats(npy_path: Path, check_monotonic: bool) -> dict[str, Any]:
    events = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"expected [N, 4] array, got {events.shape}")
    if events.shape[0] == 0:
        raise ValueError("empty event window")

    t_col = events[:, 0]
    is_monotonic = True
    if check_monotonic and events.shape[0] > 1:
        is_monotonic = bool(np.all(t_col[1:] >= t_col[:-1]))

    return {
        "num_events": int(events.shape[0]),
        "window_start_us": int(t_col[0]),
        "window_end_us": int(t_col[-1]),
        "duration_us": int(t_col[-1] - t_col[0]),
        "event_timestamps_monotonic": is_monotonic,
    }


def reject_row(part: DatasetPart, npy_path: Path | None, xml_path: Path | None, reason: str) -> dict[str, Any]:
    return {
        "subset": part.subset,
        "split": part.split,
        "npy_path": str(npy_path) if npy_path is not None else "",
        "xml_path": str(xml_path) if xml_path is not None else "",
        "reason": reason,
    }


def collect_part(
    part: DatasetPart,
    *,
    check_monotonic: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    if not part.numpy_dir.is_dir():
        rejected.append(reject_row(part, None, None, f"missing numpy dir: {part.numpy_dir}"))
        return accepted, rejected
    if not part.xml_dir.is_dir():
        rejected.append(reject_row(part, None, None, f"missing xml dir: {part.xml_dir}"))
        return accepted, rejected

    npy_files = sorted(part.numpy_dir.glob("*.npy"), key=lambda path: (frame_number(path), path.name))
    xml_stems = {path.stem: path for path in part.xml_dir.glob("*.xml")}
    npy_stems = {path.stem for path in npy_files}

    for xml_stem, xml_path in sorted(xml_stems.items()):
        if xml_stem not in npy_stems:
            rejected.append(reject_row(part, None, xml_path, "xml_without_matching_npy"))

    for npy_path in npy_files:
        xml_path = xml_stems.get(npy_path.stem)
        if xml_path is None:
            rejected.append(reject_row(part, npy_path, None, "npy_without_matching_xml"))
            continue

        try:
            event_stats = load_event_stats(npy_path, check_monotonic)
            annotation_stats = parse_voc_xml(xml_path)
        except Exception as exc:  # noqa: BLE001 - manifest needs exact reject reason.
            rejected.append(
                reject_row(part, npy_path, xml_path, f"{type(exc).__name__}: {str(exc)}")
            )
            continue

        accepted.append(
            {
                "subset": part.subset,
                "split": part.split,
                "frame_stem": npy_path.stem,
                "frame_number": frame_number(npy_path),
                "npy_path": str(npy_path),
                "xml_path": str(xml_path),
                **event_stats,
                **annotation_stats,
            }
        )

    return accepted, rejected


def assign_sequences(
    rows: list[dict[str, Any]],
    *,
    max_sequence_gap_us: int,
) -> None:
    by_part: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_part[(row["subset"], row["split"])].append(row)

    for (subset, split), part_rows in sorted(by_part.items()):
        part_rows.sort(key=lambda row: (row["frame_number"], row["frame_stem"]))

        sequence_index = -1
        position = 0
        previous: dict[str, Any] | None = None

        for row in part_rows:
            gap_us: int | None = None
            frame_number_gap: int | None = None
            break_reason = ""
            is_sequence_start = False

            if previous is None:
                is_sequence_start = True
                break_reason = "first_accepted_window"
            else:
                gap_us = row["window_start_us"] - previous["window_end_us"]
                frame_number_gap = row["frame_number"] - previous["frame_number"]
                if frame_number_gap <= 0:
                    is_sequence_start = True
                    break_reason = "non_increasing_frame_number"
                elif gap_us < 0:
                    is_sequence_start = True
                    break_reason = "negative_timestamp_gap"
                elif gap_us > max_sequence_gap_us:
                    is_sequence_start = True
                    break_reason = f"timestamp_gap_gt_{max_sequence_gap_us}us"

            if is_sequence_start:
                sequence_index += 1
                position = 0
            else:
                position += 1

            row["sequence_index"] = sequence_index
            row["sequence_id"] = f"{subset}_{split}_seq{sequence_index:04d}"
            row["sequence_position"] = position
            row["is_sequence_start"] = int(is_sequence_start)
            row["sequence_break_reason"] = break_reason
            row["previous_accepted_frame_stem"] = previous["frame_stem"] if previous else ""
            row["gap_from_previous_us"] = "" if gap_us is None else gap_us
            row["delta_t_s"] = "" if gap_us is None else gap_us / 1_000_000.0
            row["frame_number_gap_from_previous"] = "" if frame_number_gap is None else frame_number_gap
            row["has_missing_or_rejected_between"] = (
                "" if frame_number_gap is None else int(frame_number_gap > 1)
            )

            previous = row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "subset",
        "split",
        "sequence_id",
        "sequence_index",
        "sequence_position",
        "is_sequence_start",
        "sequence_break_reason",
        "frame_stem",
        "frame_number",
        "previous_accepted_frame_stem",
        "frame_number_gap_from_previous",
        "has_missing_or_rejected_between",
        "gap_from_previous_us",
        "delta_t_s",
        "window_start_us",
        "window_end_us",
        "duration_us",
        "num_events",
        "event_timestamps_monotonic",
        "width",
        "height",
        "depth",
        "num_boxes",
        "class_counts_json",
        "boxes_json",
        "has_invalid_boxes",
        "invalid_boxes_json",
        "npy_path",
        "xml_path",
    ]
    ordered = [name for name in preferred if name in fieldnames] + [
        name for name in fieldnames if name not in preferred
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    root: Path,
    max_sequence_gap_us: int,
    check_monotonic: bool,
) -> dict[str, Any]:
    part_summary: dict[str, Any] = {}
    sequence_lengths: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()

    for row in rows:
        part_key = f"{row['subset']}/{row['split']}"
        part = part_summary.setdefault(
            part_key,
            {
                "accepted_windows": 0,
                "sequence_count": 0,
                "total_events": 0,
                "total_boxes": 0,
                "empty_windows": 0,
                "duration_us_min": None,
                "duration_us_max": None,
                "num_events_min": None,
                "num_events_max": None,
                "sequence_break_reasons": Counter(),
                "object_count_histogram": Counter(),
            },
        )
        part["accepted_windows"] += 1
        part["total_events"] += row["num_events"]
        part["total_boxes"] += row["num_boxes"]
        part["empty_windows"] += int(row["num_boxes"] == 0)
        part["duration_us_min"] = row["duration_us"] if part["duration_us_min"] is None else min(part["duration_us_min"], row["duration_us"])
        part["duration_us_max"] = row["duration_us"] if part["duration_us_max"] is None else max(part["duration_us_max"], row["duration_us"])
        part["num_events_min"] = row["num_events"] if part["num_events_min"] is None else min(part["num_events_min"], row["num_events"])
        part["num_events_max"] = row["num_events"] if part["num_events_max"] is None else max(part["num_events_max"], row["num_events"])
        if row["is_sequence_start"]:
            part["sequence_count"] += 1
            if row["sequence_break_reason"]:
                part["sequence_break_reasons"][row["sequence_break_reason"]] += 1
        part["object_count_histogram"][str(row["num_boxes"])] += 1
        sequence_lengths[row["sequence_id"]] += 1
        object_counts[str(row["num_boxes"])] += 1

    for part in part_summary.values():
        part["sequence_break_reasons"] = dict(part["sequence_break_reasons"])
        part["object_count_histogram"] = dict(part["object_count_histogram"])

    return {
        "dataset_root": str(root),
        "max_sequence_gap_us": max_sequence_gap_us,
        "check_monotonic": check_monotonic,
        "accepted_windows": len(rows),
        "rejected_windows": len(rejected),
        "sequence_count": len(sequence_lengths),
        "sequence_length_min": min(sequence_lengths.values()) if sequence_lengths else 0,
        "sequence_length_max": max(sequence_lengths.values()) if sequence_lengths else 0,
        "sequence_length_median": float(np.median(list(sequence_lengths.values())))
        if sequence_lengths
        else 0.0,
        "object_count_histogram": dict(object_counts),
        "parts": part_summary,
        "rejected": rejected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/birb/grvc/PEDRo_original"),
        help="PEDRo_original dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manifests"),
        help="Directory for generated manifest files.",
    )
    parser.add_argument(
        "--max-sequence-gap-us",
        type=int,
        default=1_000_000,
        help="Reset inferred memory sequence when timestamp gap exceeds this value.",
    )
    parser.add_argument(
        "--skip-monotonic-check",
        action="store_true",
        help="Do not scan each timestamp column for monotonicity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    check_monotonic = not args.skip_monotonic_check

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for part in dataset_parts(root):
        part_accepted, part_rejected = collect_part(part, check_monotonic=check_monotonic)
        accepted.extend(part_accepted)
        rejected.extend(part_rejected)

    assign_sequences(accepted, max_sequence_gap_us=args.max_sequence_gap_us)
    accepted.sort(
        key=lambda row: (
            row["subset"],
            row["split"],
            row["sequence_index"],
            row["sequence_position"],
            row["frame_number"],
        )
    )

    manifest_path = output_dir / "pedro_original_manifest.csv"
    rejected_path = output_dir / "pedro_original_rejected.csv"
    summary_path = output_dir / "pedro_original_manifest_summary.json"

    write_csv(manifest_path, accepted)
    write_csv(rejected_path, rejected)

    summary = build_summary(
        accepted,
        rejected,
        root=root,
        max_sequence_gap_us=args.max_sequence_gap_us,
        check_monotonic=check_monotonic,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"accepted_windows={len(accepted)}")
    print(f"rejected_windows={len(rejected)}")
    print(f"sequence_count={summary['sequence_count']}")
    print(f"manifest={manifest_path}")
    print(f"rejected={rejected_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
