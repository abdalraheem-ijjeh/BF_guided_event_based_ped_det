# Uncertainty-Aware Event Memory for EventFPN-ConvGRU Pedestrian Detection

## 1. Overview

This project explores a temporally aware event-based pedestrian detector built with:

- **DAVIS346 event-camera data**
- **PEDRo dataset**
- **EventFPN**
- **ConvGRU**
- a pedestrian detection head

The proposed extension is an **uncertainty-aware event memory** that generates soft spatial priors for the detector.

Implementation note: the current code is a causal **detection-memory prior** with heuristic uncertainty. It does not perform motion compensation, identity association, optical/event-flow warping, or calibrated Bayesian uncertainty estimation. Those are future extensions, not properties of the current training pipeline.

The memory is not a conventional post-processing tracker. Instead of only smoothing detections after inference, it maintains a persistent belief about where pedestrian evidence may continue to exist, how reliable that belief is, and how uncertain each spatial hypothesis has become over time.

The detector receives the current event representation together with memory-generated prior maps.

```text
Past detections + current event support
        ↓
Heuristic uncertainty-aware detection memory
        ↓
Spatial belief / uncertainty / age priors
        ↓
Current event representation + prior maps
        ↓
EventFPN + ConvGRU
        ↓
Current pedestrian detections
        ↓
Memory update for the next event window
```

The principal objective is still **pedestrian detection enhancement**, especially when event evidence is sparse, intermittent, ambiguous, or temporarily absent.

---

## 2. Motivation

Event cameras report brightness changes rather than dense image frames. This creates detection failure modes that are different from standard RGB or frame-based pedestrian detection:

- a slowly moving pedestrian may generate weak events;
- a temporarily stationary pedestrian may almost disappear;
- event density may vary strongly between windows;
- background motion may dominate the event stream;
- short occlusion can interrupt detections;
- detector confidence may fluctuate across adjacent windows.

A detector that only sees the current event window may miss pedestrians that remain physically present but generate weak event evidence.

The proposed memory addresses this by carrying forward a spatial belief from previous windows, while explicitly modeling whether that belief should be trusted.

In one sentence:

```text
The memory gives the event-based detector a causal, uncertainty-aware prior about where pedestrian evidence is likely to persist.
```

---

## 3. Core Research Idea

The key change from a traditional filtering or tracking approach is that the persistent state is represented as **event-conditioned spatial memory**, not only as a list of tracked bounding boxes.

The full research direction may store maps or object-linked map components such as:

- pedestrian belief;
- uncertainty;
- reliability;
- age or staleness;
- event support;
- optional local motion consistency.

The current implementation stores full-frame belief, heuristic uncertainty, age, and event-support maps aligned with the detector tensor.

The method can still use detections to initialize, reinforce, or suppress memory hypotheses, but its primary output is a set of **soft spatial priors**, not track IDs.

---

## 4. Recommended Architecture

The recommended first implementation uses:

- full-frame event input;
- an uncertainty-aware memory bank;
- memory-generated soft prior maps;
- no hard spatial cropping;
- no raw-event rejection outside predicted regions;
- a retrained or fine-tuned EventFPN-ConvGRU detector;
- causal sequential training;
- fixed handcrafted memory dynamics in the first stage;
- optional learnable memory-update parameters only after a fixed baseline is validated.

The memory should participate in both training and deployment data flow. The detector must learn how to interpret the memory channels during training.

---

## 5. Complete Design Flow

```text
                              DAVIS346 event stream
                                       │
                                       ▼
                          Timestamped event ring buffer
                                       │
                                       ▼
                  Construct current event representation E_k
             voxel grid / time surface / multi-window representation
                                       │
                                       │
Previous memory state                  │
{B_(k-1), U_(k-1), A_(k-1), S_(k-1)}   │
                  │                    │
                  ▼                    │
        Temporal memory propagation    │
    decay / diffusion / motion shift   │
                  │                    │
                  ▼                    │
       Event-conditioned memory update │
 event support / silence / confidence │
                  │                    │
                  ▼                    │
         Prior-map rasterization       │
 belief / uncertainty / age / support │
                  │                    │
                  └──────────┬─────────┘
                             ▼
               Concatenate event and prior channels
                       X_k = [E_k, M_k]
                             │
                             ▼
                          EventFPN
                             │
                             ▼
                          ConvGRU
                             │
                             ▼
                       Detection head
                             │
                             ▼
              Pedestrian boxes and confidence scores
                             │
                             ▼
                    Detection-informed update
                             │
                             ▼
                  Updated memory for k+1
```

---

## 6. Memory State

For each timestamped event window \(k\), maintain a memory state:

\[
\mathcal{M}_k =
\{B_k, U_k, A_k, S_k\}
\]

where:

- \(B_k \in [0,1]^{H \times W}\): pedestrian belief map;
- \(U_k \in [0,1]^{H \times W}\): uncertainty map;
- \(A_k \in [0,1]^{H \times W}\): age or staleness map;
- \(S_k \in [0,1]^{H \times W}\): recent event-support map.

For DAVIS346:

\[
H = 260,
\qquad
W = 346
\]

The first implementation can store full-frame maps. A later optimized version can store sparse object-centered memory components and rasterize them into full-frame maps only before detector inference.

---

## 7. Event Support

The memory should distinguish between:

- areas with fresh event evidence;
- areas with weak but plausible evidence;
- areas with no recent event support;
- areas where absence of events is expected;
- areas where absence of events contradicts the current belief.

A simple event-support map can be computed from the current event representation:

\[
S_k(u,v) =
\operatorname{clip}
\left(
\frac{N_k(u,v)}{N_{\mathrm{ref}}},
0,
1
\right)
\]

where \(N_k(u,v)\) is a local event count or event-energy measure. In the current detector-integrated implementation, the default support source is the normalized count channel of each temporal stage, not a sum over all heterogeneous event channels.

The support map does not directly decide detection. It modulates memory reliability and uncertainty.

---

## 8. Temporal Memory Propagation

Before using the current event window, propagate the previous memory forward:

\[
\bar{B}_k =
\alpha_B B_{k-1}
\]

\[
\bar{U}_k =
\operatorname{clip}
\left(
U_{k-1} + \alpha_U \Delta t_k,
0,
1
\right)
\]

\[
\bar{A}_k =
\operatorname{clip}
\left(
A_{k-1} + \alpha_A \Delta t_k,
0,
1
\right)
\]

where \(\Delta t_k\) is computed from event-window timestamps:

\[
\Delta t_k =
t_k^{\mathrm{window\ end}}
-
t_{k-1}^{\mathrm{window\ end}}
\]

Do not use inference completion time or CPU callback intervals as the memory timestep.

The first version uses simple in-place decay and uncertainty growth. It does not move memory according to pedestrian motion. Later versions may add event-flow-based memory warping, learned propagation, or object-level velocity estimates.

---

## 9. Detection-Informed Memory Update

Current detections reinforce memory. For a detected pedestrian box \(d_j\), create a soft box or Gaussian support map \(D_j(u,v)\). Combine all detections:

\[
D_k(u,v) =
\max_j D_j(u,v)
\]

Then update belief:

\[
B_k =
\operatorname{clip}
\left(
\beta_B \bar{B}_k
+
(1-\beta_B)D_k,
0,
1
\right)
\]

Update uncertainty:

\[
U_k =
\operatorname{clip}
\left(
\bar{U}_k(1 - \eta_D D_k)
+
\eta_S(1-S_k)\bar{B}_k,
0,
1
\right)
\]

This means:

- detections reduce uncertainty locally;
- missing event support can increase uncertainty where belief remains high;
- stale belief gradually becomes less reliable;
- the detector can still override memory because the priors are soft.

---

## 10. Event-Silence Awareness

Event silence is not always negative evidence.

A pedestrian who slows down or stops may generate few events, but may still be present. Therefore, the memory should not immediately erase pedestrian belief just because the current event count is low.

Recommended first policy:

- low event support increases uncertainty;
- repeated low support increases age;
- belief decays slowly instead of disappearing immediately;
- detections reset uncertainty and age locally;
- very stale and uncertain memory is suppressed.

This is a central event-camera-specific part of the method. The system should learn that absence of events can mean either disappearance or temporary low motion, depending on temporal context.

---

## 11. Prior Maps

The detector input should include memory priors such as:

1. belief map \(M_{\mathrm{bel}}\);
2. uncertainty map \(M_{\mathrm{unc}}\);
3. age map \(M_{\mathrm{age}}\);
4. event-support map \(M_{\mathrm{sup}}\).

The recommended first input is:

\[
\mathbf{X}_k =
\operatorname{concat}
\left(
\mathbf{E}_k,
M_{\mathrm{bel},k},
M_{\mathrm{unc},k},
M_{\mathrm{age},k},
M_{\mathrm{sup},k}
\right)
\]

Example:

```python
# event_tensor: [B, C_event, 260, 346]
# prior_tensor: [B, 4,       260, 346]

model_input = torch.cat(
    [event_tensor, prior_tensor],
    dim=1,
)
```

The first EventFPN layer must be changed to accept the additional channels.

The network must then be retrained or fine-tuned.

---

## 12. Why the Priors Must Be Soft

The memory priors must not be used to discard full-frame event information.

Do not:

- remove events outside high-belief regions;
- process only remembered pedestrian regions;
- hard-mask the background;
- crop exclusively around memory hypotheses.

Hard masking would prevent the detector from discovering:

- newly entering pedestrians;
- pedestrians after memory failure;
- pedestrians following unexpected motion;
- pedestrians outside all existing priors.

The detector must be able to:

- use reliable memory;
- ignore uncertain memory;
- override wrong memory;
- detect a pedestrian without any memory prior.

---

## 13. Object-Centered Memory Option

The simplest design stores full-frame memory maps. A more efficient and more interpretable variant stores object-centered memory components:

\[
h_i =
\{c_x,c_y,w,h,e_i,u_i,a_i,r_i\}
\]

where:

- \(c_x,c_y,w,h\): spatial support of the hypothesis;
- \(e_i\): latent event-memory embedding;
- \(u_i\): uncertainty;
- \(a_i\): age;
- \(r_i\): reliability.

Each hypothesis is rasterized into full-frame prior maps before inference. This keeps the detector interface unchanged while allowing memory to be managed sparsely.

This object-centered version resembles internal tracking, but the final purpose remains detection improvement through prior generation.

---

## 14. Relation to Kalman and Particle Filters

A Kalman filter or particle filter can be viewed as a special case of a temporal prior generator.

However, the recommended framing is broader:

```text
Temporal state estimator
        ↓
uncertainty-aware spatial priors
        ↓
event-based detector
```

A linear Kalman filter provides one Gaussian hypothesis per object. A particle filter provides multiple weighted hypotheses per object. The proposed event memory instead maintains spatial belief and uncertainty fields that can be updated by event support, event silence, and detector confidence.

This makes the approach more event-native than a traditional tracker because the memory is not only propagated from box kinematics; it is also conditioned on the structure and absence of events.

---

## 15. Training Strategy

The memory must be present during both training and deployment.

During training:

1. process event windows in temporal order;
2. propagate previous memory to timestamp \(t_k\);
3. build priors using only information available before or within the current causal window;
4. run EventFPN-ConvGRU on current events plus memory priors;
5. compute detection loss;
6. update memory using detections, ground-truth-assisted signals, or scheduled sampling;
7. continue to the next timestamp.

Do not train the detector with event-only tensors and append memory channels only at deployment.

Do not construct the current prior directly from the current ground-truth box before inference. That leaks the answer into the input.

---

## 16. Teacher Forcing and Scheduled Sampling

At the beginning of training, the memory can be updated using noisy previous ground-truth boxes:

\[
\tilde{\mathbf{z}}_{k-1}
=
\mathbf{z}_{k-1}^{\mathrm{GT}}
+
\boldsymbol{\epsilon}
\]

Then gradually replace teacher-forced memory updates with detector-generated updates.

Recommended sequence:

1. noisy previous ground-truth boxes;
2. mixture of ground truth and predictions;
3. mostly predicted detections;
4. fully online detector-generated memory updates.

This reduces the train-deployment mismatch.

---

## 17. Memory Corruption

Perfect memory creates overdependence.

During training, randomly simulate:

- shifted belief regions;
- inflated uncertainty;
- missing memory regions;
- false memory regions;
- stale memory;
- weak event support;
- noisy event support;
- complete memory dropout.

For complete memory dropout:

\[
M_k = 0
\]

for a random percentage of training windows.

The detector must remain functional when no valid memory prior exists.

---

## 18. Optional Learnable Memory Dynamics

Only after the fixed memory version demonstrates improvement should memory dynamics be made learnable.

Potential learnable components:

- belief decay rate;
- uncertainty growth rate;
- event-support weighting;
- detection-confidence weighting;
- age suppression;
- memory update gates;
- small convolutional memory-update module.

A learnable update can be written as:

\[
\mathcal{M}_k =
g_{\phi}
\left(
\mathcal{M}_{k-1},
\mathbf{E}_k,
\mathbf{D}_k,
\Delta t_k
\right)
\]

where \(g_{\phi}\) should remain lightweight enough for real-time execution.

Avoid making the first version a large unconstrained recurrent module. Start with interpretable fixed dynamics, then learn small structured update parameters.

---

## 19. Real-Time Requirement

The complete runtime must satisfy:

\[
T_{\mathrm{repr}}
+
T_{\mathrm{memory}}
+
T_{\mathrm{prior}}
+
T_{\mathrm{network}}
+
T_{\mathrm{post}}
<
T_{\mathrm{stride}}
\]

where:

- \(T_{\mathrm{repr}}\): event representation;
- \(T_{\mathrm{memory}}\): memory propagation and update;
- \(T_{\mathrm{prior}}\): prior-map generation;
- \(T_{\mathrm{network}}\): EventFPN-ConvGRU inference;
- \(T_{\mathrm{post}}\): decoding, NMS, and memory-management logic.

For a 40 ms stride:

\[
T_{\mathrm{total}} < 40 \text{ ms}
\]

For a 10 ms stride:

\[
T_{\mathrm{total}} < 10 \text{ ms}
\]

The stride, not the window duration, determines the computational budget.

---

## 20. Computational Guidance

Avoid Python loops over all image pixels.

Recommended implementation choices:

- keep memory maps as PyTorch tensors;
- update full-frame maps with vectorized operations;
- rasterize object-centered hypotheses only inside bounded regions;
- avoid repeated CPU-GPU copies;
- keep event representation, prior maps, and network input on the same device;
- profile memory-update time separately from network inference time.

If using object-centered memory, generate each local prior only inside a bounded support region rather than over the entire sensor.

---

## 21. Operations to Avoid

Do not use:

- one memory update per raw event in Python;
- one state per pixel with expensive per-pixel control flow;
- hard memory-guided cropping;
- raw-event filtering outside remembered areas;
- synchronous camera acquisition and inference;
- repeated CPU-to-GPU copies;
- a large fully learned memory network in the first version;
- training priors that use current ground truth before inference.

The memory should guide detection, not constrain what the detector is allowed to see.

---

## 22. Evaluation Metrics

Because detection improvement is the main objective, prioritize:

- AP;
- AP\(_{50}\);
- AP\(_{75}\);
- precision;
- recall;
- false-positive rate;
- false-negative rate;
- localization error.

Evaluate difficult event-camera conditions separately:

- low event density;
- slow pedestrian motion;
- stationary pedestrians;
- rapid camera motion;
- partial occlusion;
- crowded scenes;
- new pedestrian entry;
- intermittent detections;
- high background event activity.

---

## 23. Memory Diagnostics

Inspect:

- belief-map quality;
- uncertainty calibration;
- age-map behavior;
- event-support response;
- memory persistence during low-event intervals;
- false memory persistence;
- missed pedestrian recovery;
- detector dependence on priors;
- behavior under complete memory dropout.

Optional tracking-style diagnostics may still be useful:

- ID switches;
- fragmentation;
- hypothesis lifetime;
- association failures.

These are diagnostics only. The main output remains pedestrian detections.

---

## 24. Real-Time Metrics

Measure:

- event-representation time;
- memory propagation time;
- memory update time;
- prior-map generation time;
- neural-network inference time;
- decoding and NMS time;
- end-to-end latency;
- dropped events;
- event-buffer backlog.

Report:

\[
T_{50},
\quad
T_{95},
\quad
T_{99}
\]

A robust real-time system should satisfy:

\[
T_{99}
<
T_{\mathrm{stride}}
\]

not only:

\[
\operatorname{mean}(T)
<
T_{\mathrm{stride}}
\]

---

## 25. Ablation Study

| Configuration | Event input | Memory information | Learnable memory |
|---|---|---|---|
| A | Original event tensor | None | No |
| B | Original event tensor | Belief map | No |
| C | Original event tensor | Belief + uncertainty + age + support | No |
| D | Original event tensor | Same as C with memory corruption training | No |
| E | Original event tensor | Same as C | Structured learnable update gates |
| F | Original event tensor | Object-centered memory priors | Optional |
| G | Motion-compensated events | Uncertainty-aware memory priors | Optional |

The most important initial comparison is:

\[
\text{A versus C}
\]

Only when C clearly outperforms A should learnable memory dynamics be introduced.

---

## 26. Recommended Development Phases

### Phase 1 - Baseline profiling

Measure the original EventFPN-ConvGRU system:

- AP;
- recall;
- latency;
- memory;
- event-buffer behavior;
- performance under low-event-density windows.

### Phase 2 - Fixed full-frame memory maps

Implement:

- belief map;
- uncertainty map;
- age map;
- event-support map;
- timestamp-based propagation;
- detection-informed update;
- memory reset at sequence boundaries.

Do not yet make the memory learnable.

### Phase 3 - Detector input integration

Add memory prior channels to the EventFPN input.

Retrain or fine-tune the detector using temporally ordered sequences.

### Phase 4 - Robust sequential training

Add:

- teacher forcing;
- scheduled sampling;
- memory corruption;
- false memory regions;
- missing memory regions;
- complete memory dropout.

### Phase 5 - Memory calibration

Tune:

- belief decay;
- uncertainty growth;
- age growth;
- event-support scaling;
- detection-confidence weighting;
- stale-memory suppression threshold.

### Phase 6 - Learnable structured memory

Make selected memory update rates or gates learnable.

Compare against the fixed version.

### Phase 7 - Optional object-centered memory

Replace full-frame memory storage with sparse object-centered hypotheses while preserving the same prior-map interface.

### Phase 8 - Optional motion-compensated memory

Use event-derived local motion or ego-motion compensation to shift memory forward before generating priors.

---

## 27. Final Recommended Pipeline

\[
\boxed{
\begin{aligned}
&\text{DAVIS346 events}
\rightarrow
\text{event representation}
\\
&\text{previous memory}
\rightarrow
\text{uncertainty-aware propagation}
\rightarrow
\text{soft spatial priors}
\\
&[\text{event representation},\text{memory priors}]
\rightarrow
\text{EventFPN}
\rightarrow
\text{ConvGRU}
\rightarrow
\text{pedestrian detections}
\\
&\text{detections + event support}
\rightarrow
\text{memory update}
\rightarrow
\text{next inference cycle}
\end{aligned}
}
\]

---

## 28. Final Recommendation

The first implementation should use:

- fixed uncertainty-aware memory dynamics;
- belief, uncertainty, age, and event-support prior maps;
- full-frame event input;
- vectorized memory updates;
- timestamp-based propagation;
- causal sequential training;
- memory corruption and dropout;
- no hard cropping;
- no raw-event filtering;
- no large learned memory module in the first version.

The memory must be present during training and deployment.

The neural network learns how to use the memory-generated priors.

Only after the fixed memory architecture demonstrates a clear detection benefit should structured learnable memory dynamics be introduced.

---

## 29. Technical Summary

The most accurate statement of the method is:

\[
\boxed{
\text{An uncertainty-aware event memory generates causal spatial priors to enhance future pedestrian detection inference.}
}
\]

The recommended research sequence is:

\[
\boxed{
\text{Fixed event memory priors}
\rightarrow
\text{validate detection improvement}
\rightarrow
\text{calibrate memory dynamics}
\rightarrow
\text{optionally learn structured update gates}
}
\]

This design provides a practical balance among:

- event-camera specificity;
- temporal persistence;
- uncertainty awareness;
- real-time feasibility;
- compatibility with EventFPN and ConvGRU;
- controlled ablation;
- robustness to sparse, intermittent, or ambiguous event observations.
