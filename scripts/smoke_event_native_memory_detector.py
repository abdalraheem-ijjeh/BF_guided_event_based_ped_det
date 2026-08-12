from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

THIS_REPO = Path(__file__).resolve().parents[1]
if str(THIS_REPO) not in sys.path:
    sys.path.append(str(THIS_REPO))

from small_object_approach.small_object_losses import compute_small_object_dense_detection_loss  # noqa: E402
from models.event_native_detector import EventNativeRecurrentDetector  # noqa: E402

from src.event_memory import EventMemoryConfig, UncertaintyAwareEventMemory
from src.event_native_adapter import (  # noqa: E402
    ManifestEventNativeSequenceDataset,
    SequentialEventNativeBatchSampler,
    append_priors_to_stage_tensors,
    collate_event_native_memory,
    event_native_config_from_dict,
    load_checkpoint_with_expanded_input,
)


DEFAULT_CONFIG = Path("configs/event_native_memory_pedro_p2_centerness_lite96_crop224.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test event-native detector with memory priors.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pedro_original_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default="main")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def build_model(cfg: dict) -> EventNativeRecurrentDetector:
    return EventNativeRecurrentDetector(
        in_channels=int(cfg["model"]["in_channels"]),
        stem_channels=int(cfg["model"]["stem_channels"]),
        stage_channels=cfg["model"]["stage_channels"],
        fpn_channels=int(cfg["model"]["fpn_channels"]),
        gru_hidden_channels=int(cfg["model"]["gru_hidden_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        num_feature_levels=int(cfg["model"]["num_feature_levels"]),
        use_centerness=bool(cfg["model"].get("use_centerness", False)),
        dropout=float(cfg["model"]["dropout"]),
    )


def teacher_forcing_boxes(targets: list[dict[str, torch.Tensor | str]], device: torch.device) -> torch.Tensor:
    max_boxes = max((int(target["boxes"].shape[0]) for target in targets), default=0)
    if max_boxes == 0:
        return torch.zeros((len(targets), 0, 5), dtype=torch.float32, device=device)

    boxes = torch.zeros((len(targets), max_boxes, 5), dtype=torch.float32, device=device)
    for batch_idx, target in enumerate(targets):
        target_boxes = target["boxes"].to(device=device, dtype=torch.float32)
        count = int(target_boxes.shape[0])
        if count == 0:
            continue
        boxes[batch_idx, :count, :4] = target_boxes
        boxes[batch_idx, :count, 4] = 1.0
    return boxes


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    detector_cfg = event_native_config_from_dict(cfg)

    if detector_cfg.event_plus_prior_channels != int(cfg["model"]["in_channels"]):
        raise ValueError(
            f"config model.in_channels={cfg['model']['in_channels']} does not match "
            f"event-plus-prior channels={detector_cfg.event_plus_prior_channels}"
        )

    memory_cfg = EventMemoryConfig(
        height=int(detector_cfg.image_size),
        width=int(detector_cfg.image_size),
    )
    model_cfg = dict(cfg)
    model_cfg["model"] = dict(cfg["model"])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = ManifestEventNativeSequenceDataset(
        args.manifest,
        detector_cfg,
        split=args.split,
        subset=args.subset,
        include_empty=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=SequentialEventNativeBatchSampler(dataset, batch_size=1),
        collate_fn=collate_event_native_memory,
        num_workers=0,
    )

    model = build_model(model_cfg).to(device)
    if args.checkpoint:
        load_report = load_checkpoint_with_expanded_input(model, args.checkpoint, device=device)
        print(f"[INFO] checkpoint_load={load_report}")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
    memory = UncertaintyAwareEventMemory(memory_cfg, batch_size=1, device=device)
    loss_cfg = cfg["loss"]
    small_loss_cfg = loss_cfg.get("small_object", {})

    print(
        "[INFO] smoke_setup "
        f"dataset_rows={len(dataset)} sequences={len(dataset.sequence_ids)} "
        f"event_channels={detector_cfg.event_channels} "
        f"event_plus_prior_channels={detector_cfg.event_plus_prior_channels} "
        f"image_size={detector_cfg.image_size} device={device}"
    )

    completed = 0
    for stage_tensors, targets, metas in loader:
        meta = metas[0]
        if bool(meta["is_sequence_start"]):
            memory.reset(batch_size=1)

        stage_tensors = [stage.to(device) for stage in stage_tensors]
        priors = memory.make_priors(
            stage_tensors[-1],
            window_end_time=float(meta["window_end_time"]),
        )
        stage_tensors_with_priors = append_priors_to_stage_tensors(
            stage_tensors,
            priors,
            image_size=int(detector_cfg.image_size),
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(stage_tensors_with_priors)
        loss_out = compute_small_object_dense_detection_loss(
            outputs=outputs.detections,
            targets=targets,
            image_size=int(detector_cfg.image_size),
            cls_weight=loss_cfg["cls_weight"],
            box_weight=loss_cfg["box_weight"],
            giou_weight=loss_cfg["giou_weight"],
            centerness_weight=float(loss_cfg.get("centerness_weight", 0.0)),
            focal_alpha=loss_cfg["focal_alpha"],
            focal_gamma=loss_cfg["focal_gamma"],
            min_positive_radius=int(loss_cfg.get("min_positive_radius", 0)),
            max_positive_radius=int(loss_cfg.get("max_positive_radius", 1)),
            small_area_threshold=float(small_loss_cfg.get("area_threshold", 1024.0)),
            small_height_threshold=float(small_loss_cfg.get("height_threshold", 45.0)),
            small_object_weight=float(small_loss_cfg.get("weight", 2.5)),
            assign_adjacent_level=bool(small_loss_cfg.get("assign_adjacent_level", True)),
            adjacent_max_sqrt_area=float(small_loss_cfg.get("adjacent_max_sqrt_area", 48.0)),
            small_center_weight=float(small_loss_cfg.get("center_weight", 0.0)),
            small_size_weight=float(small_loss_cfg.get("size_weight", 0.0)),
            small_center_error_cap=float(small_loss_cfg.get("center_error_cap", 4.0)),
            small_size_error_cap=float(small_loss_cfg.get("size_error_cap", 2.0)),
        )
        loss_out.loss.backward()
        optimizer.step()

        memory.update_with_detections(teacher_forcing_boxes(targets, device=device))

        completed += 1
        print(
            "[INFO] step "
            f"{completed}/{args.steps} seq={meta['sequence_id']} pos={meta['sequence_position']} "
            f"start={meta['is_sequence_start']} loss={float(loss_out.loss.item()):.4f} "
            f"prior_shape={tuple(priors.shape)} stage_shape={tuple(stage_tensors_with_priors[-1].shape)}"
        )
        if completed >= int(args.steps):
            break

    if completed == 0:
        raise RuntimeError("no batches were processed")
    print("[INFO] smoke_event_native_memory_detector passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
