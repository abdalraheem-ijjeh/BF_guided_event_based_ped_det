"""Dense detection loss with explicit small-object emphasis.

This module is intentionally separate from ``training/losses.py`` so the
current training pipeline remains unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.heads.dense_det_head import DenseDetOutput
from training.losses import LossOutput, _decode_ltrb_map, _generalized_iou, _level_for_box


def _is_small_box(
    box_norm: torch.Tensor,
    image_size: int,
    area_threshold: float,
    height_threshold: float,
) -> bool:
    width = float((box_norm[2] - box_norm[0]).clamp(min=0.0).item()) * float(image_size)
    height = float((box_norm[3] - box_norm[1]).clamp(min=0.0).item()) * float(image_size)
    area = width * height
    return area <= float(area_threshold) or height <= float(height_threshold)


def _levels_for_box(
    box_norm: torch.Tensor,
    image_size: int,
    num_levels: int,
    is_small: bool,
    assign_adjacent: bool,
    adjacent_max_sqrt_area: float,
) -> List[int]:
    primary = _level_for_box(box_norm, image_size=image_size, num_levels=num_levels)
    if not (is_small and assign_adjacent and num_levels >= 4):
        return [primary]

    wh = (box_norm[2:] - box_norm[:2]).clamp(min=0.0)
    sqrt_area = float(torch.sqrt((wh[0] * wh[1]).clamp(min=1e-6)).item()) * float(image_size)
    if sqrt_area > float(adjacent_max_sqrt_area):
        return [primary]

    # For tiny objects, supervise both P2 and P3. This gives the detector two
    # nearby scales to learn from without changing the model architecture.
    return sorted({primary, 0, min(1, num_levels - 1)})


def _build_small_object_dense_targets(
    targets: Sequence[Dict[str, torch.Tensor | str]],
    level_shapes: Sequence[Tuple[int, int]],
    image_size: int,
    min_positive_radius: int,
    max_positive_radius: int,
    small_area_threshold: float,
    small_height_threshold: float,
    small_object_weight: float,
    assign_adjacent_level: bool,
    adjacent_max_sqrt_area: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    batch_size = len(targets)
    num_levels = len(level_shapes)
    cls_targets = []
    box_targets = []
    pos_masks = []
    center_targets = []
    pos_weights = []
    for h, w in level_shapes:
        cls_targets.append(torch.zeros((batch_size, 1, h, w), dtype=torch.float32))
        box_targets.append(torch.zeros((batch_size, 4, h, w), dtype=torch.float32))
        pos_masks.append(torch.zeros((batch_size, 1, h, w), dtype=torch.bool))
        center_targets.append(torch.zeros((batch_size, 1, h, w), dtype=torch.float32))
        pos_weights.append(torch.ones((batch_size, 1, h, w), dtype=torch.float32))

    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]
        if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
            continue
        boxes_norm = boxes / float(image_size)
        for box in boxes_norm:
            is_small = _is_small_box(
                box,
                image_size=image_size,
                area_threshold=small_area_threshold,
                height_threshold=small_height_threshold,
            )
            object_weight = float(small_object_weight) if is_small else 1.0
            levels = _levels_for_box(
                box,
                image_size=image_size,
                num_levels=num_levels,
                is_small=is_small,
                assign_adjacent=assign_adjacent_level,
                adjacent_max_sqrt_area=adjacent_max_sqrt_area,
            )

            for level_idx in levels:
                h, w = level_shapes[level_idx]
                cx = float(((box[0] + box[2]) * 0.5).item()) * float(w)
                cy = float(((box[1] + box[3]) * 0.5).item()) * float(h)
                gx_center = min(w - 1, max(0, int(cx)))
                gy_center = min(h - 1, max(0, int(cy)))
                box_w = float((box[2] - box[0]).item()) * float(w)
                box_h = float((box[3] - box[1]).item()) * float(h)
                radius_x = max(int(min_positive_radius), min(int(max_positive_radius), int(box_w / 4.0)))
                radius_y = max(int(min_positive_radius), min(int(max_positive_radius), int(box_h / 4.0)))

                for gy in range(max(0, gy_center - radius_y), min(h, gy_center + radius_y + 1)):
                    for gx in range(max(0, gx_center - radius_x), min(w, gx_center + radius_x + 1)):
                        cell_cx = (float(gx) + 0.5) / float(w)
                        cell_cy = (float(gy) + 0.5) / float(h)
                        left = cell_cx - float(box[0].item())
                        top = cell_cy - float(box[1].item())
                        right = float(box[2].item()) - cell_cx
                        bottom = float(box[3].item()) - cell_cy
                        if min(left, top, right, bottom) <= 0.0:
                            continue
                        lr_min = min(left, right)
                        lr_max = max(left, right)
                        tb_min = min(top, bottom)
                        tb_max = max(top, bottom)
                        centerness = ((lr_min / max(lr_max, 1e-6)) * (tb_min / max(tb_max, 1e-6))) ** 0.5

                        cls_targets[level_idx][batch_idx, 0, gy, gx] = 1.0
                        box_targets[level_idx][batch_idx, :, gy, gx] = torch.tensor(
                            [left, top, right, bottom], dtype=torch.float32
                        )
                        pos_masks[level_idx][batch_idx, 0, gy, gx] = True
                        center_targets[level_idx][batch_idx, 0, gy, gx] = float(centerness)
                        pos_weights[level_idx][batch_idx, 0, gy, gx] = max(
                            float(pos_weights[level_idx][batch_idx, 0, gy, gx].item()),
                            object_weight,
                        )

    return cls_targets, box_targets, pos_masks, center_targets, pos_weights


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp(min=1e-6)


def compute_small_object_dense_detection_loss(
    outputs: DenseDetOutput,
    targets: Sequence[Dict[str, torch.Tensor | str]],
    image_size: int,
    cls_weight: float = 1.0,
    box_weight: float = 5.0,
    giou_weight: float = 2.0,
    centerness_weight: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    min_positive_radius: int = 0,
    max_positive_radius: int = 1,
    small_area_threshold: float = 1024.0,
    small_height_threshold: float = 45.0,
    small_object_weight: float = 2.5,
    assign_adjacent_level: bool = True,
    adjacent_max_sqrt_area: float = 48.0,
    small_center_weight: float = 0.0,
    small_size_weight: float = 0.0,
    small_center_error_cap: float = 4.0,
    small_size_error_cap: float = 2.0,
) -> LossOutput:
    small_object_weight = max(1.0, float(small_object_weight))
    level_shapes = [(int(logit.shape[-2]), int(logit.shape[-1])) for logit in outputs.logits]
    cls_targets, box_targets, pos_masks, center_targets, pos_weights = _build_small_object_dense_targets(
        targets,
        level_shapes=level_shapes,
        image_size=image_size,
        min_positive_radius=int(min_positive_radius),
        max_positive_radius=int(max_positive_radius),
        small_area_threshold=float(small_area_threshold),
        small_height_threshold=float(small_height_threshold),
        small_object_weight=float(small_object_weight),
        assign_adjacent_level=bool(assign_adjacent_level),
        adjacent_max_sqrt_area=float(adjacent_max_sqrt_area),
    )

    device = outputs.logits[0].device
    cls_loss_total = torch.zeros((), device=device)
    box_loss_total = torch.zeros((), device=device)
    giou_loss_total = torch.zeros((), device=device)
    center_loss_total = torch.zeros((), device=device)
    small_center_loss_total = torch.zeros((), device=device)
    small_size_loss_total = torch.zeros((), device=device)
    total_pos = 0
    weighted_pos_total = 0.0
    small_pos_total = 0
    small_loc_pos_total = 0

    for level_idx, (logits, box_preds) in enumerate(zip(outputs.logits, outputs.boxes)):
        cls_t = cls_targets[level_idx].to(device=device)
        box_t = box_targets[level_idx].to(device=device)
        pos_mask = pos_masks[level_idx].to(device=device)
        center_t = center_targets[level_idx].to(device=device)
        weight_t = pos_weights[level_idx].to(device=device)
        decoded_boxes = _decode_ltrb_map(box_preds)
        decoded_targets = _decode_ltrb_map(box_t)

        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, cls_t, reduction="none")
        p_t = probs * cls_t + (1.0 - probs) * (1.0 - cls_t)
        alpha_t = focal_alpha * cls_t + (1.0 - focal_alpha) * (1.0 - cls_t)
        focal = alpha_t * ((1.0 - p_t).pow(focal_gamma)) * ce
        cls_weight_map = torch.ones_like(focal)
        cls_weight_map[pos_mask] = weight_t[pos_mask]
        cls_loss_total = cls_loss_total + (focal * cls_weight_map).mean()

        pos_expand = pos_mask.expand_as(box_preds)
        if bool(pos_mask.any()):
            pred_ltrb = box_preds[pos_expand].view(-1, 4)
            true_ltrb = box_t[pos_expand].view(-1, 4)
            weights = weight_t[pos_mask].view(-1)
            l1_per_box = F.l1_loss(pred_ltrb, true_ltrb, reduction="none").mean(dim=1)
            box_loss_total = box_loss_total + _weighted_mean(l1_per_box, weights)

            pred_boxes = decoded_boxes[pos_expand].view(-1, 4)
            true_boxes = decoded_targets[pos_expand].view(-1, 4)
            giou = _generalized_iou(pred_boxes, true_boxes)
            giou_loss_total = giou_loss_total + _weighted_mean(1.0 - giou, weights)

            small_mask = weights > 1.0
            if bool(small_mask.any()) and (float(small_center_weight) > 0.0 or float(small_size_weight) > 0.0):
                pred_small = pred_boxes[small_mask]
                true_small = true_boxes[small_mask]
                small_weights = weights[small_mask]
                true_wh = (true_small[:, 2:] - true_small[:, :2]).clamp(min=2.0 / float(image_size))
                pred_wh = (pred_small[:, 2:] - pred_small[:, :2]).clamp(min=2.0 / float(image_size))

                if float(small_center_weight) > 0.0:
                    pred_center = 0.5 * (pred_small[:, :2] + pred_small[:, 2:])
                    true_center = 0.5 * (true_small[:, :2] + true_small[:, 2:])
                    center_error = ((pred_center - true_center).abs() / true_wh).clamp(
                        max=float(small_center_error_cap)
                    )
                    center_per_box = F.smooth_l1_loss(
                        center_error,
                        torch.zeros_like(center_error),
                        reduction="none",
                        beta=0.5,
                    ).mean(dim=1)
                    small_center_loss_total = small_center_loss_total + _weighted_mean(center_per_box, small_weights)

                if float(small_size_weight) > 0.0:
                    size_error = torch.log(pred_wh / true_wh).abs().clamp(max=float(small_size_error_cap))
                    size_per_box = F.smooth_l1_loss(
                        size_error,
                        torch.zeros_like(size_error),
                        reduction="none",
                        beta=0.5,
                    ).mean(dim=1)
                    small_size_loss_total = small_size_loss_total + _weighted_mean(size_per_box, small_weights)

                small_loc_pos_total += int(small_mask.sum().item())

            if outputs.centerness is not None and float(centerness_weight) > 0.0:
                center_logits = outputs.centerness[level_idx]
                center_loss = F.binary_cross_entropy_with_logits(
                    center_logits[pos_mask],
                    center_t[pos_mask],
                    reduction="none",
                )
                center_loss_total = center_loss_total + _weighted_mean(center_loss, weights)

            total_pos += int(pos_mask.sum().item())
            weighted_pos_total += float(weights.sum().detach().item())
            small_pos_total += int((weights > 1.0).sum().item())

    loss = (
        cls_weight * cls_loss_total
        + box_weight * box_loss_total
        + giou_weight * giou_loss_total
        + float(centerness_weight) * center_loss_total
        + float(small_center_weight) * small_center_loss_total
        + float(small_size_weight) * small_size_loss_total
    )
    return LossOutput(
        loss=loss,
        metrics={
            "loss": float(loss.detach().item()),
            "loss_cls": float(cls_loss_total.detach().item()),
            "loss_box": float(box_loss_total.detach().item()),
            "loss_giou": float(giou_loss_total.detach().item()),
            "loss_centerness": float(center_loss_total.detach().item()),
            "loss_small_center": float(small_center_loss_total.detach().item()),
            "loss_small_size": float(small_size_loss_total.detach().item()),
            "num_pos": float(total_pos),
            "num_small_pos": float(small_pos_total),
            "num_small_loc_pos": float(small_loc_pos_total),
            "weighted_pos": float(weighted_pos_total),
        },
    )
