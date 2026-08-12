"""Losses for the first dense event-native detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.heads.dense_det_head import DenseDetOutput


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0.0)
    return wh[:, 0] * wh[:, 1]


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def _generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    cx1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    cy1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    cx2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    cy2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    c_area = (cx2 - cx1).clamp(min=0.0) * (cy2 - cy1).clamp(min=0.0)
    return iou - (c_area - union.clamp(min=1e-6)) / c_area.clamp(min=1e-6)


def _level_for_box(box: torch.Tensor, image_size: int, num_levels: int) -> int:
    wh = (box[2:] - box[:2]).clamp(min=0.0)
    scale = float(torch.sqrt((wh[0] * wh[1]).clamp(min=1e-6)).item()) * float(image_size)
    if num_levels <= 1:
        return 0
    if num_levels == 2:
        return 0 if scale < 64.0 else 1
    if num_levels >= 4:
        if scale < 32.0:
            return 0
        if scale < 64.0:
            return 1
        if scale < 128.0:
            return 2
        return min(3, num_levels - 1)
    if scale < 48.0:
        return 0
    if scale < 96.0:
        return 1
    return 2


def _cell_centers(level_h: int, level_w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = (torch.arange(level_h, device=device, dtype=torch.float32) + 0.5) / float(level_h)
    xs = (torch.arange(level_w, device=device, dtype=torch.float32) + 0.5) / float(level_w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def _decode_ltrb_map(box_map: torch.Tensor) -> torch.Tensor:
    if box_map.ndim != 4:
        raise ValueError(f"Expected (B,4,H,W), got {tuple(box_map.shape)}")
    batch, _, level_h, level_w = box_map.shape
    device = box_map.device
    grid_x, grid_y = _cell_centers(level_h, level_w, device=device)
    grid_x = grid_x.view(1, 1, level_h, level_w).expand(batch, 1, level_h, level_w)
    grid_y = grid_y.view(1, 1, level_h, level_w).expand(batch, 1, level_h, level_w)
    left = box_map[:, 0:1]
    top = box_map[:, 1:2]
    right = box_map[:, 2:3]
    bottom = box_map[:, 3:4]
    x1 = (grid_x - left).clamp(min=0.0, max=1.0)
    y1 = (grid_y - top).clamp(min=0.0, max=1.0)
    x2 = (grid_x + right).clamp(min=0.0, max=1.0)
    y2 = (grid_y + bottom).clamp(min=0.0, max=1.0)
    return torch.cat([x1, y1, x2, y2], dim=1)


def _build_dense_targets(
    targets: Sequence[Dict[str, torch.Tensor | str]],
    level_shapes: Sequence[Tuple[int, int]],
    image_size: int,
    min_positive_radius: int = 0,
    max_positive_radius: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    batch_size = len(targets)
    num_levels = len(level_shapes)
    cls_targets = []
    box_targets = []
    pos_masks = []
    center_targets = []
    for h, w in level_shapes:
        cls_targets.append(torch.zeros((batch_size, 1, h, w), dtype=torch.float32))
        box_targets.append(torch.zeros((batch_size, 4, h, w), dtype=torch.float32))
        pos_masks.append(torch.zeros((batch_size, 1, h, w), dtype=torch.bool))
        center_targets.append(torch.zeros((batch_size, 1, h, w), dtype=torch.float32))

    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]
        if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
            continue
        boxes_norm = boxes / float(image_size)
        for box in boxes_norm:
            level_idx = _level_for_box(box, image_size=image_size, num_levels=num_levels)
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
    return cls_targets, box_targets, pos_masks, center_targets


def compute_dense_detection_loss(
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
) -> LossOutput:
    level_shapes = [(int(logit.shape[-2]), int(logit.shape[-1])) for logit in outputs.logits]
    cls_targets, box_targets, pos_masks, center_targets = _build_dense_targets(
        targets,
        level_shapes=level_shapes,
        image_size=image_size,
        min_positive_radius=int(min_positive_radius),
        max_positive_radius=int(max_positive_radius),
    )

    device = outputs.logits[0].device
    cls_loss_total = torch.zeros((), device=device)
    box_loss_total = torch.zeros((), device=device)
    giou_loss_total = torch.zeros((), device=device)
    center_loss_total = torch.zeros((), device=device)
    total_pos = 0

    for level_idx, (logits, box_preds) in enumerate(zip(outputs.logits, outputs.boxes)):
        cls_t = cls_targets[level_idx].to(device=device)
        box_t = box_targets[level_idx].to(device=device)
        pos_mask = pos_masks[level_idx].to(device=device)
        center_t = center_targets[level_idx].to(device=device)
        decoded_boxes = _decode_ltrb_map(box_preds)
        decoded_targets = _decode_ltrb_map(box_t)

        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, cls_t, reduction="none")
        p_t = probs * cls_t + (1.0 - probs) * (1.0 - cls_t)
        alpha_t = focal_alpha * cls_t + (1.0 - focal_alpha) * (1.0 - cls_t)
        focal = alpha_t * ((1.0 - p_t).pow(focal_gamma)) * ce
        cls_loss_total = cls_loss_total + focal.mean()

        pos_expand = pos_mask.expand_as(box_preds)
        if bool(pos_mask.any()):
            pred_ltrb = box_preds[pos_expand].view(-1, 4)
            true_ltrb = box_t[pos_expand].view(-1, 4)
            box_loss_total = box_loss_total + F.l1_loss(pred_ltrb, true_ltrb, reduction="mean")
            pred_boxes = decoded_boxes[pos_expand].view(-1, 4)
            true_boxes = decoded_targets[pos_expand].view(-1, 4)
            giou = _generalized_iou(pred_boxes, true_boxes)
            giou_loss_total = giou_loss_total + (1.0 - giou).mean()
            if outputs.centerness is not None and float(centerness_weight) > 0.0:
                center_logits = outputs.centerness[level_idx]
                center_loss_total = center_loss_total + F.binary_cross_entropy_with_logits(
                    center_logits[pos_mask],
                    center_t[pos_mask],
                    reduction="mean",
                )
            total_pos += int(pos_mask.sum().item())

    loss = (
        cls_weight * cls_loss_total
        + box_weight * box_loss_total
        + giou_weight * giou_loss_total
        + float(centerness_weight) * center_loss_total
    )
    return LossOutput(
        loss=loss,
        metrics={
            "loss": float(loss.detach().item()),
            "loss_cls": float(cls_loss_total.detach().item()),
            "loss_box": float(box_loss_total.detach().item()),
            "loss_giou": float(giou_loss_total.detach().item()),
            "loss_centerness": float(center_loss_total.detach().item()),
            "num_pos": float(total_pos),
        },
    )
