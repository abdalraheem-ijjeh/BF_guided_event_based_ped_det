"""Top-level event-native recurrent detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn

from models.backbones.event_fpn import EventFPNBackbone
from models.heads.dense_det_head import DenseDetHead, DenseDetOutput
from models.temporal.multiscale_convgru import MultiScaleConvGRU


@dataclass
class EventNativeDetectorOutput:
    temporal_features: List[torch.Tensor]
    detections: DenseDetOutput


class EventNativeRecurrentDetector(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        stem_channels: int = 32,
        stage_channels: Sequence[int] = (64, 128, 192),
        fpn_channels: int = 128,
        gru_hidden_channels: int = 128,
        num_classes: int = 1,
        num_feature_levels: int = 3,
        use_centerness: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_feature_levels = int(num_feature_levels)
        self.use_centerness = bool(use_centerness)
        self.backbone = EventFPNBackbone(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stage_channels=tuple(stage_channels),
            fpn_channels=fpn_channels,
            num_feature_levels=self.num_feature_levels,
            dropout=dropout,
        )
        self.temporal = MultiScaleConvGRU(
            input_channels=fpn_channels,
            hidden_channels=gru_hidden_channels,
            num_feature_levels=self.num_feature_levels,
        )
        self.head = DenseDetHead(
            in_channels=gru_hidden_channels,
            num_classes=num_classes,
            num_feature_levels=self.num_feature_levels,
            use_centerness=self.use_centerness,
        )

    def forward(self, stage_tensors: Sequence[torch.Tensor]) -> EventNativeDetectorOutput:
        if len(stage_tensors) == 0:
            raise ValueError("stage_tensors must contain at least one temporal stage")

        feature_sequence = [self.backbone(stage_tensor) for stage_tensor in stage_tensors]
        temporal_features = self.temporal(feature_sequence)
        detections = self.head(temporal_features)
        return EventNativeDetectorOutput(
            temporal_features=temporal_features,
            detections=detections,
        )
