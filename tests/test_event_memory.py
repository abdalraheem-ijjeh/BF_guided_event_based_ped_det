import torch

from src.event_memory import EventMemoryConfig, UncertaintyAwareEventMemory


def test_make_priors_returns_four_channels() -> None:
    cfg = EventMemoryConfig(height=8, width=10, support_pool_kernel=1)
    memory = UncertaintyAwareEventMemory(cfg, batch_size=2)
    events = torch.zeros(2, 3, 8, 10)

    priors = memory.make_priors(events, window_end_time=1.0)

    assert priors.shape == (2, 4, 8, 10)
    assert torch.allclose(priors[:, 0], torch.zeros(2, 8, 10))
    assert torch.allclose(priors[:, 1], torch.zeros(2, 8, 10))
    assert torch.allclose(priors[:, 2], torch.zeros(2, 8, 10))
    assert torch.allclose(priors[:, 3], torch.zeros(2, 8, 10))


def test_event_support_uses_current_event_energy() -> None:
    cfg = EventMemoryConfig(height=4, width=5, event_count_reference=2.0, support_pool_kernel=1)
    memory = UncertaintyAwareEventMemory(cfg)
    events = torch.zeros(1, 2, 4, 5)
    events[0, :, 1, 2] = torch.tensor([1.0, -1.0])

    priors = memory.make_priors(events)

    assert priors[0, 3, 1, 2].item() == 1.0
    assert priors[0, 0, 1, 2].item() == 0.0


def test_detection_update_reinforces_belief_and_reduces_uncertainty() -> None:
    cfg = EventMemoryConfig(height=6, width=6, support_pool_kernel=1, belief_mix=0.5)
    memory = UncertaintyAwareEventMemory(cfg)
    events = torch.zeros(1, 1, 6, 6)
    memory.make_priors(events)

    boxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0, 0.8]]])
    memory.update_with_detections(boxes)

    assert memory.belief[0, 0, 2, 2].item() > 0.0
    assert memory.uncertainty[0, 0, 2, 2].item() < 1.0
    assert memory.age[0, 0, 2, 2].item() < 1.0


def test_new_detection_does_not_create_silence_uncertainty_from_new_belief() -> None:
    cfg = EventMemoryConfig(
        height=6,
        width=6,
        support_pool_kernel=1,
        belief_mix=0.5,
        detection_uncertainty_reduction=0.85,
        silence_uncertainty_growth=0.5,
    )
    memory = UncertaintyAwareEventMemory(cfg)
    events = torch.zeros(1, 1, 6, 6)
    memory.make_priors(events)

    boxes = torch.tensor([[[1.0, 1.0, 3.0, 3.0, 1.0]]])
    memory.update_with_detections(boxes)

    assert memory.belief[0, 0, 2, 2].item() > 0.0
    assert memory.uncertainty[0, 0, 2, 2].item() == 0.0


def test_timestamp_propagation_decays_belief_and_grows_uncertainty() -> None:
    cfg = EventMemoryConfig(
        height=4,
        width=4,
        support_pool_kernel=1,
        belief_decay_per_second=0.5,
        uncertainty_growth_per_second=0.5,
        age_growth_per_second=0.5,
    )
    memory = UncertaintyAwareEventMemory(cfg)
    memory.belief.fill_(1.0)
    memory.uncertainty.zero_()
    memory.age.zero_()
    events = torch.zeros(1, 1, 4, 4)

    memory.make_priors(events, window_end_time=1.0)
    memory.make_priors(events, window_end_time=2.0)

    assert torch.allclose(memory.belief, torch.full_like(memory.belief, 0.5))
    assert torch.allclose(memory.uncertainty, torch.full_like(memory.uncertainty, 0.5))
    assert torch.allclose(memory.age, torch.full_like(memory.age, 0.5))


def test_large_absolute_timestamps_preserve_subsecond_delta() -> None:
    cfg = EventMemoryConfig(
        height=2,
        width=2,
        support_pool_kernel=1,
        belief_decay_per_second=0.5,
        uncertainty_growth_per_second=0.5,
        age_growth_per_second=0.5,
    )
    memory = UncertaintyAwareEventMemory(cfg)
    memory.belief.fill_(1.0)
    memory.uncertainty.zero_()
    memory.age.zero_()
    events = torch.zeros(1, 1, 2, 2)

    memory.make_priors(events, window_end_time=1_673_194_919.701658)
    memory.make_priors(events, window_end_time=1_673_194_920.101658)

    assert torch.allclose(memory.belief, torch.full_like(memory.belief, 0.8))
    assert torch.allclose(memory.uncertainty, torch.full_like(memory.uncertainty, 0.2))
    assert torch.allclose(memory.age, torch.full_like(memory.age, 0.2))


def test_reset_clears_sequence_state() -> None:
    cfg = EventMemoryConfig(height=3, width=3)
    memory = UncertaintyAwareEventMemory(cfg)
    memory.belief.fill_(0.7)
    memory.uncertainty.zero_()

    memory.reset(batch_size=2)

    assert memory.belief.shape == (2, 1, 3, 3)
    assert memory.belief.max().item() == 0.0
    assert memory.uncertainty.max().item() == 0.0


def test_uncertainty_and_age_are_zero_without_active_belief() -> None:
    cfg = EventMemoryConfig(height=4, width=4, support_pool_kernel=1)
    memory = UncertaintyAwareEventMemory(cfg)
    events = torch.zeros(1, 1, 4, 4)

    memory.make_priors(events, window_end_time=1.0)
    priors = memory.make_priors(events, window_end_time=2.0)

    assert priors[:, 0].max().item() == 0.0
    assert priors[:, 1].max().item() == 0.0
    assert priors[:, 2].max().item() == 0.0
