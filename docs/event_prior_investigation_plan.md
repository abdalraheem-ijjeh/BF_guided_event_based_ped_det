# Event Plus Prior Investigation Plan

This plan targets the next research step after an event-only EventFPN-ConvGRU
baseline: compare the original event tensor against the same tensor augmented
with uncertainty-aware memory priors.

## Objective

Validate whether causal prior channels improve pedestrian detection under sparse,
intermittent, or ambiguous event evidence.

Primary comparison:

| ID | Event input | Memory priors | Memory dynamics |
| --- | --- | --- | --- |
| A | original event tensor | none | none |
| C | original event tensor | belief, uncertainty, age, support | fixed full-frame |

The first target is not to beat tracking metrics. The target is better detection:
AP, AP50, AP75, recall, false negatives, false positives, and localization error.

## Detector Interface

For each event window `k`, build:

```python
event_tensor = build_event_tensor(events_k)          # [B, C_event, H, W]
prior_tensor = memory.make_priors(event_tensor, window_end_time=t_k_end)
model_input = torch.cat([event_tensor, prior_tensor], dim=1)
```

The first EventFPN convolution must accept `C_event + 4` channels.

Do not crop, hard-mask, or discard events outside high-belief regions. The priors
are hints, not gates.

## Causal Sequential Loop

For each sequence:

1. Reset memory at the sequence boundary.
2. Process windows in timestamp order.
3. Propagate memory using event-window end timestamps.
4. Compute current event support from the current event representation.
5. Concatenate event tensor and prior tensor.
6. Run detector and compute detection loss.
7. Decode detections.
8. Update memory using detections, teacher-forced previous targets, or scheduled sampling.
9. Continue to the next timestamp.

Current ground-truth boxes must not be used to build current priors before
inference.

## Minimal Fixed Memory Baseline

Use full-frame maps at DAVIS346 resolution:

- `B`: pedestrian belief
- `U`: uncertainty
- `A`: age/staleness
- `S`: recent event support

Start with fixed dynamics:

- belief decays slowly with timestamp delta;
- uncertainty grows with timestamp delta;
- age grows with timestamp delta;
- event support is computed from current event energy/count;
- detections reinforce belief and reduce uncertainty;
- low support near high belief raises uncertainty;
- stale, highly uncertain belief is suppressed.

Use `src/event_memory.py` as the detector-agnostic reference implementation.

## Training Variants

Run these in order:

| Variant | Description | Purpose |
| --- | --- | --- |
| C0 | priors present, memory updated from predicted detections only | deployment-matched baseline |
| C1 | noisy previous ground truth for early memory updates | easier warm start |
| C2 | scheduled sampling between ground truth and predictions | reduce train/deploy mismatch |
| C3 | C2 plus memory corruption/dropout | robustness to wrong priors |

Recommended memory corruption:

- randomly shift belief regions;
- inflate uncertainty;
- remove memory regions;
- inject false belief regions;
- set all memory priors to zero for some windows;
- perturb support maps.

## Evaluation Slices

Report global detection metrics and also slice metrics by:

- low event density;
- slow pedestrian motion;
- stationary or near-stationary pedestrians;
- partial occlusion;
- rapid camera motion;
- high background activity;
- new pedestrian entry;
- windows following missed detections.

The critical failure mode to inspect is overdependence on memory. The detector
must still detect new pedestrians with empty or incorrect priors.

## Diagnostics

Save visualizations for representative sequences:

- event tensor projection;
- belief map;
- uncertainty map;
- age map;
- support map;
- predicted boxes;
- ground-truth boxes.

Inspect whether:

- belief persists through low-event intervals;
- uncertainty rises when event support disappears;
- stale false belief is eventually suppressed;
- detections recover after memory dropout;
- new pedestrians are found outside memory regions.

## Real-Time Measurements

Measure:

- event representation time;
- memory propagation time;
- prior generation time;
- network inference time;
- decoding/NMS time;
- memory update time;
- end-to-end latency.

Report `T50`, `T95`, and `T99`. For real-time feasibility, `T99` should remain
below the event-window stride.

## Integration Checklist

- Add four memory channels to the dataset/training loop.
- Expand the first EventFPN layer from `C_event` to `C_event + 4`.
- Reset memory at every sequence boundary.
- Preserve temporal order in dataloading.
- Avoid shuffling windows within a sequence.
- Keep event tensors and memory tensors on the same device.
- Train with memory present from the beginning of the prior experiment.
- Evaluate event-only and event-plus-prior models on the same sequence splits.
