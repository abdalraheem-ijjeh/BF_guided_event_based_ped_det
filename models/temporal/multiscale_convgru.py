"""Multi-scale recurrent temporal fusion.

This module fuses feature pyramids over anchor-aligned multi-dt stages using a
separate ConvGRU cell per feature scale.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        merged_channels = int(input_channels) + int(hidden_channels)
        self.hidden_channels = int(hidden_channels)

        self.gates = nn.Conv2d(
            merged_channels,
            2 * self.hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.candidate = nn.Conv2d(
            merged_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(
                (x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )

        merged = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.gates(merged))
        z, r = torch.chunk(gates, chunks=2, dim=1)

        candidate_input = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.candidate(candidate_input))
        return (1.0 - z) * h + z * h_tilde


class MultiScaleConvGRU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_feature_levels: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_feature_levels = int(num_feature_levels)

        self.cells = nn.ModuleList(
            ConvGRUCell(
                input_channels=self.input_channels,
                hidden_channels=self.hidden_channels,
                kernel_size=kernel_size,
            )
            for _ in range(self.num_feature_levels)
        )

    def forward(self, feature_sequence: Sequence[Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        if len(feature_sequence) == 0:
            raise ValueError("feature_sequence must contain at least one stage")

        hidden_states: List[torch.Tensor | None] = [None] * self.num_feature_levels

        for stage_idx, stage_features in enumerate(feature_sequence):
            if len(stage_features) != self.num_feature_levels:
                raise ValueError(
                    f"Stage {stage_idx} has {len(stage_features)} feature levels; "
                    f"expected {self.num_feature_levels}"
                )
            next_hidden: List[torch.Tensor] = []
            for level_idx, (cell, feat) in enumerate(zip(self.cells, stage_features)):
                if feat.ndim != 4:
                    raise ValueError(
                        f"Expected feature map shape (B,C,H,W) at stage {stage_idx}, "
                        f"level {level_idx}; got {tuple(feat.shape)}"
                    )
                next_hidden.append(cell(feat, hidden_states[level_idx]))
            hidden_states = next_hidden

        return [state for state in hidden_states if state is not None]
