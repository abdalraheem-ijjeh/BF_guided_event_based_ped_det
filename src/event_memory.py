from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EventMemoryConfig:
    height: int = 260
    width: int = 346
    belief_decay_per_second: float = 0.25
    uncertainty_growth_per_second: float = 0.35
    age_growth_per_second: float = 0.50
    belief_mix: float = 0.80
    detection_uncertainty_reduction: float = 0.85
    silence_uncertainty_growth: float = 0.20
    event_count_reference: float = 4.0
    support_pool_kernel: int = 5
    stale_age_threshold: float = 0.85
    stale_uncertainty_threshold: float = 0.85
    stale_suppression: float = 0.15
    min_detection_score: float = 0.05
    active_belief_epsilon: float = 1e-4


class UncertaintyAwareEventMemory:
    """Full-frame uncertainty-aware memory for event-based detection priors.

    The memory exposes four prior channels:
    belief, uncertainty, age/staleness, and current event support.

    Use `make_priors` before detector inference, then call `update_with_detections`
    after decoding the detector output. Current ground truth should not be used to
    create priors for the same timestep.
    """

    def __init__(
        self,
        config: EventMemoryConfig | None = None,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or EventMemoryConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.reset(batch_size=batch_size)

    @property
    def state(self) -> dict[str, torch.Tensor]:
        return {
            "belief": self.belief,
            "uncertainty": self.uncertainty,
            "age": self.age,
            "support": self.support,
        }

    def reset(self, batch_size: int = 1) -> None:
        shape = (batch_size, 1, self.config.height, self.config.width)
        self.belief = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.uncertainty = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.age = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.support = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self._last_window_end: torch.Tensor | None = None

    def make_priors(
        self,
        event_tensor: torch.Tensor,
        *,
        window_end_time: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propagate memory and return `[B, 4, H, W]` prior maps.

        Args:
            event_tensor: Current event representation `[B, C, H, W]`.
            window_end_time: Event-window end timestamp in seconds. This must
                come from event data, not wall-clock inference time.
        """
        self._check_event_tensor(event_tensor)
        self._match_batch_and_device(event_tensor)

        if window_end_time is not None:
            dt = self._compute_delta_t(window_end_time)
            self.propagate(dt)

        self.support = self.event_support(event_tensor)
        return torch.cat(
            [
                self.belief,
                self.uncertainty * self._active_memory_mask(),
                self.age * self._active_memory_mask(),
                self.support,
            ],
            dim=1,
        )

    def propagate(self, delta_t: float | torch.Tensor) -> None:
        dt = torch.as_tensor(delta_t, device=self.device, dtype=self.dtype).clamp_min(0)
        cfg = self.config

        belief_decay = torch.clamp(1.0 - cfg.belief_decay_per_second * dt, 0.0, 1.0)
        self.belief = (self.belief * belief_decay).clamp(0.0, 1.0)
        active = self._active_memory_mask()
        self.uncertainty = (
            self.uncertainty + cfg.uncertainty_growth_per_second * dt * active
        ).clamp(0.0, 1.0)
        self.age = (self.age + cfg.age_growth_per_second * dt * active).clamp(0.0, 1.0)

    def event_support(self, event_tensor: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        event_energy = event_tensor.detach().abs().sum(dim=1, keepdim=True)
        support = (event_energy / cfg.event_count_reference).clamp(0.0, 1.0)

        kernel = cfg.support_pool_kernel
        if kernel > 1:
            padding = kernel // 2
            support = F.avg_pool2d(support, kernel_size=kernel, stride=1, padding=padding)
        return support.clamp(0.0, 1.0)

    def update_with_detections(self, boxes: torch.Tensor) -> None:
        """Update memory after detector inference.

        Args:
            boxes: Tensor shaped `[B, N, 5]` containing `x1, y1, x2, y2, score`.
                Coordinates are pixel-space in the memory-map resolution.
        """
        detection_support = self.rasterize_boxes(boxes)
        cfg = self.config
        propagated_belief = self.belief

        self.belief = (
            cfg.belief_mix * self.belief + (1.0 - cfg.belief_mix) * detection_support
        ).clamp(0.0, 1.0)

        self.uncertainty = (
            self.uncertainty * (1.0 - cfg.detection_uncertainty_reduction * detection_support)
            + cfg.silence_uncertainty_growth * (1.0 - self.support) * propagated_belief
        ).clamp(0.0, 1.0)

        self.age = (self.age * (1.0 - detection_support)).clamp(0.0, 1.0)

        stale = (
            (self.age >= cfg.stale_age_threshold)
            & (self.uncertainty >= cfg.stale_uncertainty_threshold)
        ).to(self.dtype)
        self.belief = (
            self.belief * (1.0 - stale) + self.belief * cfg.stale_suppression * stale
        ).clamp(0.0, 1.0)

        active = self._active_memory_mask()
        self.uncertainty = self.uncertainty * active
        self.age = self.age * active

    def rasterize_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.ndim != 3 or boxes.shape[-1] != 5:
            raise ValueError("boxes must have shape [B, N, 5] with x1, y1, x2, y2, score")
        if boxes.shape[0] != self.belief.shape[0]:
            raise ValueError("boxes batch size must match memory batch size")

        boxes = boxes.to(device=self.device, dtype=self.dtype)
        cfg = self.config
        batch_size, num_boxes, _ = boxes.shape
        if num_boxes == 0:
            return torch.zeros_like(self.belief)

        x1, y1, x2, y2, score = boxes.unbind(dim=-1)
        valid = (score >= cfg.min_detection_score) & (x2 > x1) & (y2 > y1)

        xs = torch.arange(cfg.width, device=self.device, dtype=self.dtype).view(1, 1, 1, cfg.width)
        ys = torch.arange(cfg.height, device=self.device, dtype=self.dtype).view(1, 1, cfg.height, 1)

        inside_x = (xs >= x1.view(batch_size, num_boxes, 1, 1)) & (
            xs <= x2.view(batch_size, num_boxes, 1, 1)
        )
        inside_y = (ys >= y1.view(batch_size, num_boxes, 1, 1)) & (
            ys <= y2.view(batch_size, num_boxes, 1, 1)
        )
        mask = inside_x & inside_y & valid.view(batch_size, num_boxes, 1, 1)
        weighted = mask.to(self.dtype) * score.clamp(0.0, 1.0).view(batch_size, num_boxes, 1, 1)
        return weighted.amax(dim=1, keepdim=True).clamp(0.0, 1.0)

    def _compute_delta_t(self, window_end_time: float | torch.Tensor) -> torch.Tensor:
        current = torch.as_tensor(window_end_time, device=self.device, dtype=torch.float64)
        if current.ndim == 0:
            current = current.view(1)
        if self._last_window_end is None:
            delta_t = torch.zeros((), device=self.device, dtype=self.dtype)
        else:
            delta_t = (current.mean() - self._last_window_end.mean()).clamp_min(0.0).to(self.dtype)
        self._last_window_end = current.detach()
        return delta_t

    def _active_memory_mask(self) -> torch.Tensor:
        return (self.belief > float(self.config.active_belief_epsilon)).to(self.dtype)

    def _check_event_tensor(self, event_tensor: torch.Tensor) -> None:
        if event_tensor.ndim != 4:
            raise ValueError("event_tensor must have shape [B, C, H, W]")
        if event_tensor.shape[-2:] != (self.config.height, self.config.width):
            raise ValueError(
                "event_tensor spatial shape must match "
                f"({self.config.height}, {self.config.width})"
            )

    def _match_batch_and_device(self, event_tensor: torch.Tensor) -> None:
        if event_tensor.device != self.device or event_tensor.dtype != self.dtype:
            self.device = event_tensor.device
            self.dtype = event_tensor.dtype
            self.belief = self.belief.to(device=self.device, dtype=self.dtype)
            self.uncertainty = self.uncertainty.to(device=self.device, dtype=self.dtype)
            self.age = self.age.to(device=self.device, dtype=self.dtype)
            self.support = self.support.to(device=self.device, dtype=self.dtype)
            if self._last_window_end is not None:
                self._last_window_end = self._last_window_end.to(device=self.device)

        if event_tensor.shape[0] != self.belief.shape[0]:
            self.reset(batch_size=event_tensor.shape[0])
