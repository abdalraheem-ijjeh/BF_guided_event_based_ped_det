from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


THIS_REPO = Path(__file__).resolve().parents[1]
if str(THIS_REPO) not in sys.path:
    sys.path.append(str(THIS_REPO))

from models.event_native_detector import EventNativeRecurrentDetector
from small_object_approach.small_object_losses import compute_small_object_dense_detection_loss
from src.event_memory import EventMemoryConfig, UncertaintyAwareEventMemory
from src.event_native_adapter import (
    ManifestEventNativeSequenceDataset,
    SequentialEventNativeBatchSampler,
    append_priors_to_stage_tensors,
    collate_event_native_memory,
    event_native_config_from_dict,
    load_checkpoint_with_expanded_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train event-native detector with uncertainty-aware memory priors.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/event_native_memory_pedro_p2_centerness_lite96_crop224.yaml"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pedro_original_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-valid-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def freeze_batchnorm_stats(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def build_memory_config(cfg: dict, image_size: int) -> EventMemoryConfig:
    memory_cfg = cfg.get("memory", {})
    return EventMemoryConfig(
        height=int(image_size),
        width=int(image_size),
        belief_decay_per_second=float(memory_cfg.get("belief_decay_per_second", 0.25)),
        uncertainty_growth_per_second=float(memory_cfg.get("uncertainty_growth_per_second", 0.35)),
        age_growth_per_second=float(memory_cfg.get("age_growth_per_second", 0.50)),
        belief_mix=float(memory_cfg.get("belief_mix", 0.80)),
        detection_uncertainty_reduction=float(memory_cfg.get("detection_uncertainty_reduction", 0.85)),
        silence_uncertainty_growth=float(memory_cfg.get("silence_uncertainty_growth", 0.20)),
        event_count_reference=float(memory_cfg.get("event_count_reference", 4.0)),
        support_pool_kernel=int(memory_cfg.get("support_pool_kernel", 5)),
        stale_age_threshold=float(memory_cfg.get("stale_age_threshold", 0.85)),
        stale_uncertainty_threshold=float(memory_cfg.get("stale_uncertainty_threshold", 0.85)),
        stale_suppression=float(memory_cfg.get("stale_suppression", 0.15)),
        min_detection_score=float(memory_cfg.get("min_detection_score", 0.05)),
        active_belief_epsilon=float(memory_cfg.get("active_belief_epsilon", 1e-4)),
    )


def validate_config(cfg: dict) -> None:
    detector_cfg = event_native_config_from_dict(cfg)
    expected_channels = detector_cfg.event_plus_prior_channels
    if int(cfg["model"]["in_channels"]) != expected_channels:
        raise ValueError(f"model.in_channels must be {expected_channels} for event+prior training")
    if int(cfg["train"].get("batch_size", 1)) != 1:
        raise ValueError("causal memory training currently requires train.batch_size=1")
    if int(cfg.get("eval", {}).get("batch_size", 1)) != 1:
        raise ValueError("causal memory AP evaluation currently requires eval.batch_size=1")
    if int(cfg["dataset"]["anchor_dt_ms"]) != int(cfg["dataset"]["dts_ms"][-1]):
        raise ValueError("dataset.anchor_dt_ms must match the last detector stage used for support priors")
    if int(cfg.get("memory", {}).get("prior_channels", 4)) != 4:
        raise ValueError("memory.prior_channels must be 4: belief, uncertainty, age, support")


def build_loader(
    cfg: dict,
    manifest_path: Path,
    *,
    split: str,
) -> tuple[ManifestEventNativeSequenceDataset, DataLoader]:
    detector_cfg = event_native_config_from_dict(cfg)
    manifest_split = "val" if split in {"valid", "val"} else split
    dataset = ManifestEventNativeSequenceDataset(
        manifest_path,
        detector_cfg,
        split=manifest_split,
        subset=None,
        include_empty=True,
    )
    is_train = split == "train"
    sampler = SequentialEventNativeBatchSampler(
        dataset,
        batch_size=1,
        shuffle_sequences=bool(cfg["train"].get("shuffle_sequences", False)) and is_train,
        seed=int(cfg["experiment"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_event_native_memory,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=bool(cfg["train"].get("pin_memory", False)),
    )
    return dataset, loader


def teacher_forcing_probability(epoch: int, cfg: dict) -> float:
    memory_cfg = cfg.get("memory", {})
    teacher_epochs = int(memory_cfg.get("teacher_forcing_epochs", 5))
    scheduled_epochs = int(memory_cfg.get("scheduled_sampling_epochs", 15))
    if epoch < teacher_epochs:
        return 1.0
    if scheduled_epochs <= 0 or epoch >= teacher_epochs + scheduled_epochs:
        return 0.0
    progress = float(epoch - teacher_epochs + 1) / float(scheduled_epochs)
    return max(0.0, 1.0 - progress)


def boxes_from_targets(targets: list[dict[str, torch.Tensor | str]], device: torch.device) -> torch.Tensor:
    max_boxes = max((int(target["boxes"].shape[0]) for target in targets), default=0)
    if max_boxes == 0:
        return torch.zeros((len(targets), 0, 5), dtype=torch.float32, device=device)
    output = torch.zeros((len(targets), max_boxes, 5), dtype=torch.float32, device=device)
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        count = int(boxes.shape[0])
        if count:
            output[batch_idx, :count, :4] = boxes
            output[batch_idx, :count, 4] = 1.0
    return output


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[ious <= float(iou_threshold)]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def decode_memory_boxes(
    outputs,
    *,
    image_size: int,
    threshold: float,
    nms_iou: float,
    topk: int,
) -> torch.Tensor:
    batch_size = int(outputs.detections.logits[0].shape[0])
    decoded = []
    for batch_idx in range(batch_size):
        boxes_all = []
        scores_all = []
        for level_idx, (logits, box_map) in enumerate(zip(outputs.detections.logits, outputs.detections.boxes)):
            logits_i = logits[batch_idx]
            box_i = box_map[batch_idx]
            _, level_h, level_w = logits_i.shape
            ys = (torch.arange(level_h, device=logits_i.device, dtype=torch.float32) + 0.5) / float(level_h)
            xs = (torch.arange(level_w, device=logits_i.device, dtype=torch.float32) + 0.5) / float(level_w)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            scores = torch.sigmoid(logits_i).reshape(-1)
            if outputs.detections.centerness is not None:
                scores = scores * torch.sigmoid(outputs.detections.centerness[level_idx][batch_idx]).reshape(-1)
            keep = scores >= float(threshold)
            if not bool(keep.any()):
                continue
            box_ltrb = box_i.permute(1, 2, 0).reshape(-1, 4)[keep]
            centers = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=-1).reshape(-1, 4)[keep]
            scores = scores[keep]
            x1 = (centers[:, 0] - box_ltrb[:, 0]).clamp(0.0, 1.0) * float(image_size)
            y1 = (centers[:, 1] - box_ltrb[:, 1]).clamp(0.0, 1.0) * float(image_size)
            x2 = (centers[:, 2] + box_ltrb[:, 2]).clamp(0.0, 1.0) * float(image_size)
            y2 = (centers[:, 3] + box_ltrb[:, 3]).clamp(0.0, 1.0) * float(image_size)
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if bool(valid.any()):
                boxes_all.append(boxes[valid])
                scores_all.append(scores[valid])
        if not boxes_all:
            decoded.append(torch.zeros((0, 5), dtype=torch.float32, device=outputs.detections.logits[0].device))
            continue
        boxes = torch.cat(boxes_all, dim=0)
        scores = torch.cat(scores_all, dim=0)
        keep = nms_xyxy(boxes, scores, iou_threshold=float(nms_iou))
        if topk > 0:
            keep = keep[: int(topk)]
        decoded.append(torch.cat([boxes[keep], scores[keep, None]], dim=1))
    max_boxes = max((item.shape[0] for item in decoded), default=0)
    if max_boxes == 0:
        return torch.zeros((batch_size, 0, 5), dtype=torch.float32, device=outputs.detections.logits[0].device)
    padded = torch.zeros((batch_size, max_boxes, 5), dtype=torch.float32, device=outputs.detections.logits[0].device)
    for batch_idx, item in enumerate(decoded):
        if item.numel():
            padded[batch_idx, : item.shape[0]] = item
    return padded


def valid_decoded_boxes(decoded_boxes: torch.Tensor) -> list[torch.Tensor]:
    if decoded_boxes.ndim != 3 or decoded_boxes.shape[-1] != 5:
        raise ValueError(f"expected decoded boxes [B,N,5], got {tuple(decoded_boxes.shape)}")
    items = []
    for batch_idx in range(decoded_boxes.shape[0]):
        boxes = decoded_boxes[batch_idx]
        items.append(boxes[boxes[:, 4] > 0.0])
    return items


def target_boxes_for_ap(targets: list[dict[str, torch.Tensor | str]], device: torch.device) -> list[torch.Tensor]:
    return [target["boxes"].to(device=device, dtype=torch.float32) for target in targets]


def greedy_match_detections_at_iou(
    predictions: list[dict[str, np.ndarray | str]],
    ground_truths: dict[str, np.ndarray],
    *,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    total_gt = int(sum(boxes.shape[0] for boxes in ground_truths.values()))
    if not predictions:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), total_gt

    order = np.argsort([-float(prediction["score"]) for prediction in predictions])
    matched: dict[str, np.ndarray] = {
        image_key: np.zeros((boxes.shape[0],), dtype=bool)
        for image_key, boxes in ground_truths.items()
    }
    tp = np.zeros((len(order),), dtype=np.float64)
    fp = np.zeros((len(order),), dtype=np.float64)

    for rank_idx, prediction_idx in enumerate(order):
        prediction = predictions[int(prediction_idx)]
        image_key = str(prediction["image_key"])
        gt_boxes = ground_truths.get(image_key)
        pred_box = np.asarray(prediction["box"], dtype=np.float32)
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            fp[rank_idx] = 1.0
            continue

        ious = box_iou(
            torch.from_numpy(pred_box[None, :]),
            torch.from_numpy(gt_boxes.astype(np.float32, copy=False)),
        ).numpy()[0]
        available = ~matched[image_key]
        if not bool(available.any()):
            fp[rank_idx] = 1.0
            continue

        masked_ious = np.where(available, ious, -1.0)
        best_idx = int(masked_ious.argmax())
        if float(masked_ious[best_idx]) >= float(iou_threshold):
            tp[rank_idx] = 1.0
            matched[image_key][best_idx] = True
        else:
            fp[rank_idx] = 1.0

    return tp, fp, total_gt


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0 or precision.size == 0:
        return 0.0
    recall_grid = np.linspace(0.0, 1.0, 101)
    precision_at_recall = np.zeros_like(recall_grid)
    for idx, recall_level in enumerate(recall_grid):
        valid = precision[recall >= recall_level]
        precision_at_recall[idx] = valid.max() if valid.size else 0.0
    return float(precision_at_recall.mean())


def compute_detection_ap_metrics(
    predictions: list[dict[str, np.ndarray | str]],
    ground_truths: dict[str, np.ndarray],
    *,
    no_person_frame_count: int,
    no_person_thresholds: Sequence[float],
) -> dict[str, float]:
    iou_thresholds = [round(float(value), 2) for value in np.arange(0.50, 0.96, 0.05)]
    metrics: dict[str, float] = {}
    ap_values = []
    recall_values = []

    for iou_threshold in iou_thresholds:
        tp, fp, total_gt = greedy_match_detections_at_iou(
            predictions,
            ground_truths,
            iou_threshold=iou_threshold,
        )
        if total_gt == 0:
            ap = 0.0
            recall = 0.0
        else:
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall_curve = tp_cum / float(total_gt)
            precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap = interpolated_ap(recall_curve, precision_curve)
            recall = float(recall_curve[-1]) if recall_curve.size else 0.0

        metrics[f"AP{int(round(iou_threshold * 100)):02d}"] = float(ap)
        metrics[f"recall_{int(round(iou_threshold * 100)):02d}"] = float(recall)
        ap_values.append(float(ap))
        recall_values.append(float(recall))

    metrics["AP"] = float(np.mean(ap_values)) if ap_values else 0.0
    metrics["AR"] = float(np.mean(recall_values)) if recall_values else 0.0
    metrics["num_gt"] = float(sum(boxes.shape[0] for boxes in ground_truths.values()))
    metrics["num_predictions"] = float(len(predictions))

    no_person_predictions = [
        prediction
        for prediction in predictions
        if ground_truths[str(prediction["image_key"])].shape[0] == 0
    ]
    for threshold in no_person_thresholds:
        fp_count = sum(1 for prediction in no_person_predictions if float(prediction["score"]) >= float(threshold))
        key = f"no_person_fp_per_frame_thr_{int(round(float(threshold) * 100)):02d}"
        metrics[key] = float(fp_count) / float(max(1, no_person_frame_count))

    return metrics


def decoded_batch_to_ap_records(
    decoded_items: list[torch.Tensor],
    target_items: list[torch.Tensor],
    metas: list[dict],
) -> tuple[list[dict[str, np.ndarray | str]], dict[str, np.ndarray], int]:
    predictions: list[dict[str, np.ndarray | str]] = []
    ground_truths: dict[str, np.ndarray] = {}
    no_person_frames = 0

    for boxes, target_boxes, meta in zip(decoded_items, target_items, metas):
        image_key = str(meta["index"])
        target_np = target_boxes.detach().cpu().numpy().astype(np.float32, copy=False)
        ground_truths[image_key] = target_np
        if target_np.shape[0] == 0:
            no_person_frames += 1

        boxes_np = boxes.detach().cpu().numpy().astype(np.float32, copy=False)
        for box in boxes_np:
            predictions.append(
                {
                    "image_key": image_key,
                    "box": box[:4].copy(),
                    "score": float(box[4]),
                }
            )

    return predictions, ground_truths, no_person_frames


def select_memory_update_boxes(
    outputs,
    targets: list[dict[str, torch.Tensor | str]],
    *,
    device: torch.device,
    image_size: int,
    epoch: int,
    cfg: dict,
    is_train: bool,
) -> tuple[torch.Tensor, str]:
    if is_train and random.random() < teacher_forcing_probability(epoch, cfg):
        return boxes_from_targets(targets, device=device), "teacher"

    inference_cfg = cfg.get("inference", {})
    memory_cfg = cfg.get("memory", {})
    threshold = float(memory_cfg.get("prediction_update_score_threshold", inference_cfg.get("threshold", 0.25)))
    boxes = decode_memory_boxes(
        outputs,
        image_size=image_size,
        threshold=threshold,
        nms_iou=float(inference_cfg.get("nms_iou", 0.5)),
        topk=int(inference_cfg.get("topk", 50)),
    )
    return boxes.detach(), "prediction"


def maybe_dropout_priors(priors: torch.Tensor, cfg: dict, *, is_train: bool) -> torch.Tensor:
    if not is_train:
        return priors
    dropout_prob = float(cfg.get("memory", {}).get("dropout_prob", 0.0))
    if dropout_prob > 0.0 and random.random() < dropout_prob:
        return torch.zeros_like(priors)
    return priors


def compute_loss(outputs, targets, image_size: int, loss_cfg: dict):
    small_cfg = loss_cfg.get("small_object", {})
    small_enabled = bool(small_cfg.get("enabled", True))
    return compute_small_object_dense_detection_loss(
        outputs=outputs.detections,
        targets=targets,
        image_size=image_size,
        cls_weight=loss_cfg["cls_weight"],
        box_weight=loss_cfg["box_weight"],
        giou_weight=loss_cfg["giou_weight"],
        centerness_weight=float(loss_cfg.get("centerness_weight", 0.0)),
        focal_alpha=loss_cfg["focal_alpha"],
        focal_gamma=loss_cfg["focal_gamma"],
        min_positive_radius=int(loss_cfg.get("min_positive_radius", 0)),
        max_positive_radius=int(loss_cfg.get("max_positive_radius", 1)),
        small_area_threshold=float(small_cfg.get("area_threshold", 1024.0)),
        small_height_threshold=float(small_cfg.get("height_threshold", 45.0)),
        small_object_weight=float(small_cfg.get("weight", 2.5)) if small_enabled else 1.0,
        assign_adjacent_level=bool(small_cfg.get("assign_adjacent_level", True)) and small_enabled,
        adjacent_max_sqrt_area=float(small_cfg.get("adjacent_max_sqrt_area", 48.0)),
        small_center_weight=float(small_cfg.get("center_weight", 0.0)) if small_enabled else 0.0,
        small_size_weight=float(small_cfg.get("size_weight", 0.0)) if small_enabled else 0.0,
        small_center_error_cap=float(small_cfg.get("center_error_cap", 4.0)),
        small_size_error_cap=float(small_cfg.get("size_error_cap", 2.0)),
    )


def run_epoch(
    *,
    model: EventNativeRecurrentDetector,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    max_steps: int,
    split_name: str,
    collect_ap: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)
    if is_train and bool(cfg["train"].get("freeze_batchnorm", False)):
        freeze_batchnorm_stats(model)
    image_size = int(cfg["dataset"]["image_size"])
    use_amp = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    memory = UncertaintyAwareEventMemory(build_memory_config(cfg, image_size), batch_size=1, device=device)

    running: dict[str, float] = {}
    steps = 0
    teacher_updates = 0
    prediction_updates = 0
    log_every = int(cfg["train"].get("log_every", 20))
    ap_predictions: list[dict[str, np.ndarray | str]] = []
    ap_ground_truths: dict[str, np.ndarray] = {}
    ap_no_person_frames = 0
    eval_cfg = cfg.get("eval", {})
    inference_cfg = cfg.get("inference", {})

    for stage_tensors, targets, metas in loader:
        meta = metas[0]
        if bool(meta["is_sequence_start"]):
            memory.reset(batch_size=1)

        stage_tensors = [stage.to(device, non_blocking=True) for stage in stage_tensors]
        priors = memory.make_priors(stage_tensors[-1], window_end_time=float(meta["window_end_time"]))
        priors = maybe_dropout_priors(priors, cfg, is_train=is_train)
        stage_tensors = append_priors_to_stage_tensors(stage_tensors, priors, image_size=image_size)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(stage_tensors)
                loss_out = compute_loss(outputs, targets, image_size=image_size, loss_cfg=cfg["loss"])

            if is_train:
                prev_scale = scaler.get_scale()
                scaler.scale(loss_out.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["train"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= prev_scale:
                    scheduler.step()

        with torch.no_grad():
            if collect_ap:
                decoded_for_ap = decode_memory_boxes(
                    outputs,
                    image_size=image_size,
                    threshold=float(eval_cfg.get("threshold", cfg.get("inference", {}).get("threshold", 0.05))),
                    nms_iou=float(eval_cfg.get("nms_iou", cfg.get("inference", {}).get("nms_iou", 0.5))),
                    topk=int(eval_cfg.get("topk", cfg.get("inference", {}).get("topk", 100))),
                )
                decoded_items = valid_decoded_boxes(decoded_for_ap)
                target_items = target_boxes_for_ap(targets, device=device)
                batch_predictions, batch_ground_truths, batch_no_person_frames = decoded_batch_to_ap_records(
                    decoded_items,
                    target_items,
                    metas,
                )
                ap_predictions.extend(batch_predictions)
                ap_ground_truths.update(batch_ground_truths)
                ap_no_person_frames += int(batch_no_person_frames)

                memory_threshold = float(
                    cfg.get("memory", {}).get(
                        "prediction_update_score_threshold",
                        inference_cfg.get("threshold", 0.25),
                    )
                )
                update_boxes = decode_memory_boxes(
                    outputs,
                    image_size=image_size,
                    threshold=memory_threshold,
                    nms_iou=float(inference_cfg.get("nms_iou", 0.5)),
                    topk=int(inference_cfg.get("topk", 50)),
                )
                update_mode = "prediction"
            else:
                update_boxes, update_mode = select_memory_update_boxes(
                    outputs,
                    targets,
                    device=device,
                    image_size=image_size,
                    epoch=epoch,
                    cfg=cfg,
                    is_train=is_train,
                )
            memory.update_with_detections(update_boxes)
        teacher_updates += int(update_mode == "teacher")
        prediction_updates += int(update_mode == "prediction")

        for key, value in loss_out.metrics.items():
            running[key] = running.get(key, 0.0) + float(value)
        steps += 1

        if log_every > 0 and steps % log_every == 0:
            avg = {key: value / float(steps) for key, value in running.items()}
            print(
                f"[{split_name}] epoch={epoch} step={steps}/{len(loader)} "
                f"loss={avg.get('loss', 0.0):.4f} pos={avg.get('num_pos', 0.0):.1f} "
                f"teacher={teacher_updates} prediction={prediction_updates}"
            )

        if max_steps > 0 and steps >= max_steps:
            break

    if steps == 0:
        return {"loss": 0.0, "steps": 0.0}
    metrics = {key: value / float(steps) for key, value in running.items()}
    metrics["steps"] = float(steps)
    metrics["teacher_updates"] = float(teacher_updates)
    metrics["prediction_updates"] = float(prediction_updates)
    if collect_ap:
        metrics.update(
            compute_detection_ap_metrics(
                ap_predictions,
                ap_ground_truths,
                no_person_frame_count=ap_no_person_frames,
                no_person_thresholds=eval_cfg.get("no_person_thresholds", [0.2, 0.3, 0.4, 0.5]),
            )
        )
    return metrics


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, train_steps_per_epoch: int, epochs: int):
    if str(cfg["train"].get("scheduler", "")).lower() != "cosine":
        return None
    total_steps = int(epochs) * max(1, int(train_steps_per_epoch))
    warmup_steps = int(cfg["train"].get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.log_every is not None:
        cfg["train"]["log_every"] = int(args.log_every)

    set_seed(int(cfg["experiment"]["seed"]))
    validate_config(cfg)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() or cfg["train"]["device"] == "cpu" else "cpu"
    )
    output_dir = args.output_dir or Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, train_loader = build_loader(cfg, args.manifest, split="train")
    valid_ds, valid_loader = build_loader(cfg, args.manifest, split="valid")
    print(
        "[INFO] datasets "
        f"train_rows={len(train_ds)} train_sequences={len(train_ds.sequence_ids)} "
        f"valid_rows={len(valid_ds)} valid_sequences={len(valid_ds.sequence_ids)}"
    )

    model = build_model(cfg).to(device)
    if bool(cfg["train"].get("freeze_batchnorm", False)):
        print(f"[INFO] freeze_batchnorm_stats=true layers={freeze_batchnorm_stats(model)}")
    init_checkpoint = args.init_checkpoint or cfg["train"].get("init_checkpoint")
    if init_checkpoint:
        report = load_checkpoint_with_expanded_input(model, init_checkpoint, device=device)
        print(f"[INFO] init_checkpoint={init_checkpoint} report={report}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["epochs"])
    train_steps = args.max_train_steps if args.max_train_steps > 0 else len(train_loader)
    scheduler = build_scheduler(cfg, optimizer, train_steps_per_epoch=train_steps, epochs=epochs)

    log_path = output_dir / "log.jsonl"
    best_valid_loss = float("inf")
    best_valid_ap = -float("inf")
    for epoch in range(epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            cfg=cfg,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            max_steps=int(args.max_train_steps),
            split_name="train",
        )
        valid_metrics = run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            cfg=cfg,
            epoch=epoch,
            optimizer=None,
            scheduler=None,
            max_steps=int(args.max_valid_steps),
            split_name="valid",
            collect_ap=(
                int(cfg.get("eval", {}).get("real_ap_every", 1)) > 0
                and epoch % int(cfg.get("eval", {}).get("real_ap_every", 1)) == 0
            ),
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        print(
            f"[INFO] epoch={epoch} train_loss={train_metrics.get('loss', 0.0):.4f} "
            f"valid_loss={valid_metrics.get('loss', 0.0):.4f} "
            f"valid_AP={valid_metrics.get('AP', 0.0):.4f} "
            f"AP50={valid_metrics.get('AP50', 0.0):.4f} "
            f"AP75={valid_metrics.get('AP75', 0.0):.4f} "
            f"AR={valid_metrics.get('AR', 0.0):.4f} "
            f"no_person_fp50={valid_metrics.get('no_person_fp_per_frame_thr_50', 0.0):.4f} "
            f"train_teacher={train_metrics.get('teacher_updates', 0.0):.0f} "
            f"train_prediction={train_metrics.get('prediction_updates', 0.0):.0f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pth")
        valid_loss = float(valid_metrics.get("loss", float("inf")))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(checkpoint, output_dir / "checkpoint_best_loss.pth")
        valid_ap = float(valid_metrics.get("AP", -float("inf")))
        if int(args.max_valid_steps) <= 0 and valid_ap > best_valid_ap:
            best_valid_ap = valid_ap
            torch.save(checkpoint, output_dir / "checkpoint_best_ap.pth")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
