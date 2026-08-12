"""Event-native multi-scale backbone.

The first implementation is intentionally compact:

- convolutional stem for multi-channel event tensors
- residual downsampling stages
- top-down FPN outputs at three spatial scales

Input:
- tensor of shape `(B, C, H, W)`

Output:
- list of feature maps `[P3, P4, P5]` by default
- list of feature maps `[P2, P3, P4, P5]` when `num_feature_levels=4`
- each feature map has `fpn_channels` channels
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        return x * self.gate(w)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, kernel_size=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout2d(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.dropout(x)
        x = x + residual
        return self.act(x)


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = [ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2)]
        layers.extend(ResidualBlock(out_channels, dropout=dropout) for _ in range(int(num_blocks)))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class EventFPNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 32,
        stage_channels: List[int] | tuple[int, int, int] = (64, 128, 192),
        fpn_channels: int = 128,
        num_feature_levels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 3:
            raise ValueError(f"Expected exactly 3 stage_channels, got {stage_channels}")
        if int(num_feature_levels) not in (3, 4):
            raise ValueError(f"Expected num_feature_levels to be 3 or 4, got {num_feature_levels}")

        c3, c4, c5 = [int(c) for c in stage_channels]
        self.in_channels = int(in_channels)
        self.stem_channels = int(stem_channels)
        self.stage_channels = (c3, c4, c5)
        self.fpn_channels = int(fpn_channels)
        self.num_feature_levels = int(num_feature_levels)

        self.stem = nn.Sequential(
            ConvNormAct(self.in_channels, self.stem_channels, kernel_size=3, stride=2),
            ConvNormAct(self.stem_channels, self.stem_channels, kernel_size=3, stride=1),
        )

        self.stage3 = DownsampleStage(self.stem_channels, c3, num_blocks=2, dropout=dropout)
        self.stage4 = DownsampleStage(c3, c4, num_blocks=2, dropout=dropout)
        self.stage5 = DownsampleStage(c4, c5, num_blocks=2, dropout=dropout)

        self.lat3 = nn.Conv2d(c3, self.fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lat4 = nn.Conv2d(c4, self.fpn_channels, kernel_size=1, stride=1, padding=0)
        self.lat5 = nn.Conv2d(c5, self.fpn_channels, kernel_size=1, stride=1, padding=0)
        if self.num_feature_levels == 4:
            self.lat2 = nn.Conv2d(self.stem_channels, self.fpn_channels, kernel_size=1, stride=1, padding=0)

        self.out3 = ConvNormAct(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1)
        self.out4 = ConvNormAct(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1)
        self.out5 = ConvNormAct(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1)
        if self.num_feature_levels == 4:
            self.out2 = ConvNormAct(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")

        c2 = self.stem(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + torch.nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + torch.nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p3 = self.out3(p3)
        p4 = self.out4(p4)
        p5 = self.out5(p5)
        if self.num_feature_levels == 4:
            p2 = self.lat2(c2) + torch.nn.functional.interpolate(p3, size=c2.shape[-2:], mode="nearest")
            p2 = self.out2(p2)
            return [p2, p3, p4, p5]
        return [p3, p4, p5]
