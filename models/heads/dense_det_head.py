"""Dense detection head for event-native multi-scale features.

The box branch predicts normalized ``l, t, r, b`` distances from each feature
cell center to the four box edges. Distances are constrained to ``[0, 1]``
relative to the square training image size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass
class DenseDetOutput:
    logits: List[torch.Tensor]
    boxes: List[torch.Tensor]
    centerness: List[torch.Tensor] | None = None


class ScaleHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, use_centerness: bool = False) -> None:
        super().__init__()
        self.use_centerness = bool(use_centerness)
        hidden = int(in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(hidden, int(num_classes), kernel_size=1, stride=1, padding=0)
        self.box_head = nn.Conv2d(hidden, 4, kernel_size=1, stride=1, padding=0)
        if self.use_centerness:
            self.center_head = nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feat = self.stem(x)
        logits = self.cls_head(feat)
        box_raw = self.box_head(feat)
        centerness = self.center_head(feat) if self.use_centerness else None
        return logits, torch.sigmoid(box_raw), centerness


class DenseDetHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        num_feature_levels: int = 3,
        use_centerness: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.num_feature_levels = int(num_feature_levels)
        self.use_centerness = bool(use_centerness)
        self.scale_heads = nn.ModuleList(
            ScaleHead(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                use_centerness=self.use_centerness,
            )
            for _ in range(self.num_feature_levels)
        )

    def forward(self, features: Sequence[torch.Tensor]) -> DenseDetOutput:
        if len(features) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} feature maps, got {len(features)}"
            )

        logits: List[torch.Tensor] = []
        boxes: List[torch.Tensor] = []
        centerness: List[torch.Tensor] = []
        for head, feat in zip(self.scale_heads, features):
            if feat.ndim != 4:
                raise ValueError(f"Expected feature shape (B,C,H,W), got {tuple(feat.shape)}")
            cls_logits, box_map, center_map = head(feat)
            logits.append(cls_logits)
            boxes.append(box_map)
            if center_map is not None:
                centerness.append(center_map)
        return DenseDetOutput(
            logits=logits,
            boxes=boxes,
            centerness=centerness if self.use_centerness else None,
        )
