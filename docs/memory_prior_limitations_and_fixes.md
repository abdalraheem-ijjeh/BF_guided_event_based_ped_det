# Memory Prior Limitations And Fixes

This implementation should be described as a causal detection-memory prior, not
as a full motion-compensated tracker or calibrated Bayesian uncertainty model.

## Valid Limitations

- The belief map does not move. It decays in place and is reinforced by later
  detections. There is no optical flow, event flow, Kalman state, velocity, box
  association, or feature warping.
- The uncertainty map is heuristic. It is bounded, grows with elapsed time and
  low support, and is reduced by detections. It is not calibrated covariance or
  posterior probability.
- There is no identity association. Nearby pedestrians can merge into a single
  broad memory region because dense box supports are combined spatially.
- Box rasterization is rectangular and score-weighted. It is intentionally
  simple and can be replaced by Gaussian or soft masks later.

## Implementation Fixes

- Support is now computed from a configured source. The default is the detector
  count channel (`memory.support_source: count_channel`), not a heterogeneous
  sum over voxel, recency, and count channels.
- Support is now stage-specific. The 10 ms, 20 ms, and 40 ms detector stages each
  receive support computed from their own event representation. Belief,
  uncertainty, and age remain shared causal memory fields.
- `memory.prior_mode` supports ablations without changing model shape:
  `full`, `support_only`, `memory_only`, `belief_only`, and `zero`.
- Prediction updates can be support-gated with
  `memory.prediction_support_gate_strength` to reduce false-positive
  self-reinforcement in unsupported regions.
- Teacher-forced updates can be corrupted using small jitter, drops, and false
  boxes to reduce the deployment mismatch during early epochs.

## Required Ablations

At minimum, compare:

- `full`: belief, heuristic uncertainty, age, and support.
- `support_only`: current support only, no temporal memory.
- `memory_only`: temporal belief/uncertainty/age only, no current support prior.
- `zero`: 15-channel shape with all prior channels zeroed.

These ablations separate gains from the engineered support channel from gains
caused by temporal detection memory.
