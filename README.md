[README_Kalman_Guided_EventFPN_ConvGRU.md](https://github.com/user-attachments/files/30910182/README_Kalman_Guided_EventFPN_ConvGRU.md)
# Kalman-Guided EventFPN–ConvGRU Pedestrian Detection

## 1. Overview

This project extends an event-based pedestrian detector built with:

- **DAVIS346 event-camera data**
- **PEDRo dataset**
- **EventFPN**
- **ConvGRU**
- A pedestrian detection head

The proposed extension introduces a **linear Kalman filter (KF)** before each current inference step.

The KF is not used only as a conventional post-processing tracker. Instead, it maintains lightweight pedestrian states and predicts where previously detected pedestrians are likely to appear in the next event window. These predictions are converted into soft spatial prior maps and provided to the EventFPN–ConvGRU detector.

The overall concept is:

```text
Previous detections
        ↓
Kalman update
        ↓
Kalman prediction for the next timestamp
        ↓
Soft spatial prior maps
        ↓
Current event representation + prior maps
        ↓
EventFPN + ConvGRU
        ↓
Current pedestrian detections
```

The principal objective remains **pedestrian detection enhancement**, especially under sparse or intermittent event evidence.

---

## 2. Main Objective

The proposed KF integration is intended to improve:

- pedestrian recall;
- temporal consistency;
- bounding-box stability;
- detection during low-event-density intervals;
- detection of slowly moving or briefly stationary pedestrians;
- robustness to temporary missed detections;
- continuity between consecutive event windows.

The KF performs internal object-state tracking, but tracking is an enabling mechanism rather than the main output.

A precise description of the approach is:

> **Kalman-guided, tracking-assisted event-based pedestrian detection.**

---

## 3. Recommended Architecture

The recommended first implementation uses:

- a **fixed linear Kalman filter**;
- fixed transition and measurement models;
- fixed process and measurement covariance matrices;
- soft KF-generated prior maps;
- a retrained or fine-tuned EventFPN–ConvGRU detector;
- full-frame event input;
- no per-event Kalman filtering;
- no hard spatial cropping;
- no learned Kalman gain;
- no fully learned KalmanNet-style module.

The KF participates in the training and deployment data flow, but its parameters are initially frozen.

---

## 4. Complete Design Flow

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
                                       │
Previous posterior pedestrian states   │
{x_hat_(k-1)^+, P_(k-1)^+}             │
                  │                    │
                  ▼                    │
       Kalman prediction to t_k        │
       {x_hat_k^-, P_k^-}              │
                  │                    │
                  ▼                    │
         Prior-map rasterization       │
 occupancy / reliability / age maps   │
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
          Detection-to-state association and gating
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
             Matched states      Unmatched detections
                   │                   │
                   ▼                   ▼
            Kalman update       Initialize new states
                   │
                   ▼
          Updated posterior state bank
               {x_hat_k^+, P_k^+}
                   │
                   └──────────────► Used before inference at k+1
```

---

## 5. Why a Linear KF Is Recommended

For a 2-D bounding-box state, both the state transition and measurement functions are linear.

A suitable state is:

\[
\mathbf{x}_k =
\begin{bmatrix}
c_x &
c_y &
w &
h &
v_x &
v_y &
v_w &
v_h
\end{bmatrix}^{T}
\]

where:

- \(c_x, c_y\): bounding-box center;
- \(w, h\): bounding-box width and height;
- \(v_x, v_y\): center velocity;
- \(v_w, v_h\): width and height variation rates.

The detector measurement is:

\[
\mathbf{z}_k =
\begin{bmatrix}
c_x &
c_y &
w &
h
\end{bmatrix}^{T}
\]

The process model is:

\[
\mathbf{x}_k =
\mathbf{F}(\Delta t_k)\mathbf{x}_{k-1}
+
\mathbf{w}_k
\]

The measurement model is:

\[
\mathbf{z}_k =
\mathbf{H}\mathbf{x}_k
+
\mathbf{v}_k
\]

Because both models are linear, an **Extended Kalman Filter is not required** for the bounding-box state.

An EKF should only be considered for a genuinely nonlinear problem, such as:

- camera pose estimation;
- event–IMU fusion;
- quaternion orientation;
- nonlinear 3-D-to-2-D projection;
- perspective-aware motion;
- ego-motion compensation.

---

## 6. State Transition Model

For the state ordering

\[
[c_x,c_y,w,h,v_x,v_y,v_w,v_h]^T
\]

the transition matrix is:

\[
\mathbf{F}(\Delta t_k)
=
\begin{bmatrix}
\mathbf{I}_4 & \Delta t_k\mathbf{I}_4 \\
\mathbf{0}_4 & \mathbf{I}_4
\end{bmatrix}
\]

The prediction equations are:

\[
\hat{\mathbf{x}}_k^{-}
=
\mathbf{F}(\Delta t_k)\hat{\mathbf{x}}_{k-1}^{+}
\]

\[
\mathbf{P}_k^{-}
=
\mathbf{F}(\Delta t_k)
\mathbf{P}_{k-1}^{+}
\mathbf{F}^{T}(\Delta t_k)
+
\mathbf{Q}_k
\]

The measurement matrix is:

\[
\mathbf{H}
=
\begin{bmatrix}
\mathbf{I}_4 & \mathbf{0}_4
\end{bmatrix}
\]

---

## 7. Kalman Measurement Update

For a matched detection:

\[
\mathbf{y}_k
=
\mathbf{z}_k
-
\mathbf{H}\hat{\mathbf{x}}_k^{-}
\]

\[
\mathbf{S}_k
=
\mathbf{H}\mathbf{P}_k^{-}\mathbf{H}^{T}
+
\mathbf{R}_k
\]

\[
\mathbf{K}_k
=
\mathbf{P}_k^{-}\mathbf{H}^{T}\mathbf{S}_k^{-1}
\]

\[
\hat{\mathbf{x}}_k^{+}
=
\hat{\mathbf{x}}_k^{-}
+
\mathbf{K}_k\mathbf{y}_k
\]

For numerical stability, use the Joseph covariance update:

\[
\mathbf{P}_k^{+}
=
(\mathbf{I}-\mathbf{K}_k\mathbf{H})
\mathbf{P}_k^{-}
(\mathbf{I}-\mathbf{K}_k\mathbf{H})^{T}
+
\mathbf{K}_k\mathbf{R}_k\mathbf{K}_k^{T}
\]

---

## 8. Timestamp Handling

The Kalman transition interval must be calculated from event-window timestamps:

\[
\Delta t_k
=
t_k^{\mathrm{window\ end}}
-
t_{k-1}^{\mathrm{window\ end}}
\]

Do not use:

- inference completion time;
- CPU callback intervals;
- an assumed fixed frame rate when the stride varies.

For example, if two event windows end 20 ms apart:

\[
\Delta t_k = 0.020 \text{ s}
\]

even when inference takes an additional 10–15 ms.

### Window duration versus stride

- **Window duration**: the amount of event history represented.
- **Window stride**: the interval between consecutive detector calls.

The stride determines the real-time processing budget.

---

## 9. Process Noise

For one position–velocity pair, use the structured constant-acceleration covariance:

\[
\mathbf{Q}_i(\Delta t)
=
\sigma_{a,i}^{2}
\begin{bmatrix}
\frac{\Delta t^4}{4} & \frac{\Delta t^3}{2} \\
\frac{\Delta t^3}{2} & \Delta t^2
\end{bmatrix}
\]

Use separate process-noise values for:

\[
\sigma_{a,x},
\quad
\sigma_{a,y},
\quad
\sigma_{a,w},
\quad
\sigma_{a,h}
\]

This is preferable to specifying or learning every element of a dense \(8 \times 8\) matrix.

---

## 10. Measurement Noise

Initially use:

\[
\mathbf{R}
=
\operatorname{diag}
\left(
\sigma_{c_x}^{2},
\sigma_{c_y}^{2},
\sigma_w^{2},
\sigma_h^{2}
\right)
\]

These values describe detector localization uncertainty.

During the first stage:

- \(\mathbf{Q}\) is fixed;
- \(\mathbf{R}\) is fixed;
- \(\mathbf{P}_0\) is fixed;
- all KF matrices are excluded from the optimizer.

---

## 11. KF Prior Maps

A detector cannot directly consume an arbitrary list of KF state vectors. The predicted states must be converted into spatial maps aligned with the event tensor.

For DAVIS346, the standard spatial resolution is:

\[
H = 260,
\qquad
W = 346
\]

The recommended prior channels are:

1. occupancy map;
2. reliability map;
3. age map.

An optional later extension can include velocity maps.

---

## 12. Occupancy Prior

For predicted pedestrian \(j\):

\[
\boldsymbol{\mu}_j
=
\begin{bmatrix}
\hat{c}_{x,j}^{-} \\
\hat{c}_{y,j}^{-}
\end{bmatrix}
\]

Let \(\boldsymbol{\Sigma}_j\) be the positional covariance extracted from \(\mathbf{P}_j^{-}\).

Construct:

\[
G_j(u,v)
=
\exp
\left[
-\frac{1}{2}
\left(
\mathbf{r}
-
\boldsymbol{\mu}_j
\right)^{T}
\boldsymbol{\Sigma}_j^{-1}
\left(
\mathbf{r}
-
\boldsymbol{\mu}_j
\right)
\right]
\]

with:

\[
\mathbf{r}
=
\begin{bmatrix}
u \\
v
\end{bmatrix}
\]

Combine all pedestrians with:

\[
M_{\mathrm{occ}}(u,v)
=
\max_j G_j(u,v)
\]

A probabilistic union is also possible:

\[
M_{\mathrm{occ}}
=
1-\prod_j(1-G_j)
\]

The maximum operator is simpler and should be used first.

---

## 13. Reliability Prior

A track-level reliability score can be defined as:

\[
r_j
=
\exp
\left[
-\lambda_P
\operatorname{tr}(\boldsymbol{\Sigma}_j)
\right]
\exp(-\lambda_m m_j)
\]

where:

- \(\boldsymbol{\Sigma}_j\): positional covariance;
- \(m_j\): number of consecutive missed observations.

The reliability map becomes:

\[
M_{\mathrm{rel}}(u,v)
=
\max_j
\left[
r_jG_j(u,v)
\right]
\]

This map tells the detector whether a predicted region is reliable.

---

## 14. Age Prior

Define:

\[
a_j
=
\min
\left(
\frac{\tau_j}{\tau_{\max}},
1
\right)
\]

where:

- \(\tau_j\): time since the last matched detection;
- \(\tau_{\max}\): maximum retained missed duration.

Then:

\[
M_{\mathrm{age}}(u,v)
=
\max_j
\left[
a_jG_j(u,v)
\right]
\]

A low value corresponds to a recent state. A high value corresponds to a stale state.

---

## 15. Neural Network Input

The detector input becomes:

\[
\mathbf{X}_k
=
\operatorname{concat}
\left(
\mathbf{E}_k,
M_{\mathrm{occ},k},
M_{\mathrm{rel},k},
M_{\mathrm{age},k}
\right)
\]

Example:

```python
# event_tensor: [B, C_event, 260, 346]
# prior_tensor: [B, 3,       260, 346]

model_input = torch.cat(
    [event_tensor, prior_tensor],
    dim=1,
)
```

The first EventFPN layer must be changed to accept the additional channels.

The network must then be retrained or fine-tuned.

---

## 16. Why the Priors Must Be Soft

The KF priors must not be used to discard full-frame event information.

Do not:

- remove events outside predicted boxes;
- process only tracked pedestrian regions;
- hard-mask the background;
- crop exclusively around active tracks.

Hard masking would prevent the detector from discovering:

- newly entering pedestrians;
- pedestrians after track failure;
- pedestrians following unexpected motion;
- pedestrians outside all existing priors.

The detector must be able to:

- use a reliable prior;
- ignore an unreliable prior;
- override a wrong prior;
- detect a pedestrian without any prior.

---

## 17. Internal Tracking Requirements

Although the final objective is detection enhancement, the KF requires internal track-state management.

The implementation needs:

- state initialization;
- state confirmation;
- detection-to-state association;
- matched-state update;
- missed-observation handling;
- state deletion.

Thus, internal tracking is required even when track IDs are not final outputs.

---

## 18. Detection-to-State Association

Use a cost combining motion consistency and bounding-box overlap:

\[
C_{ij}
=
\lambda_M d_{M,ij}^{2}
+
\lambda_{\mathrm{IoU}}
\left(
1-\operatorname{IoU}_{ij}
\right)
\]

The squared Mahalanobis distance is:

\[
d_{M,ij}^{2}
=
\left(
\mathbf{z}_j
-
\mathbf{H}\hat{\mathbf{x}}_i^{-}
\right)^{T}
\mathbf{S}_i^{-1}
\left(
\mathbf{z}_j
-
\mathbf{H}\hat{\mathbf{x}}_i^{-}
\right)
\]

Use Mahalanobis gating before Hungarian assignment:

\[
d_{M,ij}^{2}
<
\gamma
\]

Do not use IoU alone. Event-based boxes can shift significantly between short windows.

---

## 19. Track Initialization

An unmatched detection initializes:

\[
\hat{\mathbf{x}}_0
=
[c_x,c_y,w,h,0,0,0,0]^T
\]

The initial velocity covariance should be relatively large.

A newly created state should initially be tentative.

Recommended policy:

- confirm after two or three consistent matches;
- use no prior or a weak prior while tentative;
- delete quickly when the tentative state is not confirmed.

---

## 20. Missed Observations

When a state is unmatched:

1. retain the prediction;
2. allow covariance to grow;
3. increase missed duration;
4. reduce reliability;
5. weaken its prior;
6. delete after a time-based maximum age.

Use elapsed time:

\[
\tau_{\mathrm{miss}}
=
t_k
-
t_{\mathrm{last\ matched}}
\]

instead of relying only on frame count.

---

## 21. State Deletion

Delete a state when:

- missed duration exceeds \(\tau_{\max}\);
- covariance becomes too large;
- predicted box remains outside the sensor field;
- width or height becomes invalid;
- reliability becomes too low.

A stale KF prediction must not persist indefinitely.

---

## 22. Training Strategy

### Key clarification

The recommended KF is:

- **present during training**;
- **present during deployment**;
- **not learnable initially**.

The KF participates in the training data flow because the neural detector must learn how to interpret the prior channels.

During the first stage, the optimizer updates only:

\[
\theta_{\mathrm{EventFPN}},
\quad
\theta_{\mathrm{ConvGRU}},
\quad
\theta_{\mathrm{detection\ head}}
\]

It does not update:

\[
\mathbf{F},
\quad
\mathbf{H},
\quad
\mathbf{Q},
\quad
\mathbf{R},
\quad
\mathbf{P}_0
\]

---

## 23. Training Flow

```text
Previous boxes or detections
        ↓
Fixed KF update
        ↓
Fixed KF prediction
        ↓
Occupancy / reliability / age maps
        ↓
Current event tensor + KF maps
        ↓
EventFPN + ConvGRU
        ↓
Detection head
        ↓
Detection loss
        ↓
Update neural-network weights only
```

The deployed network must receive the same type of input channels that it saw during training.

Do not train with events only and then append KF channels only at deployment.

---

## 24. Sequential Training

The training samples must preserve temporal order.

At time \(k\):

1. obtain the previous state;
2. predict it to timestamp \(t_k\);
3. create priors using only information available before \(t_k\);
4. process the current event tensor;
5. compute the current detection loss;
6. update the ConvGRU hidden state;
7. continue to the next timestamp.

Do not construct the current prior from the current ground-truth box, because that would leak the target into the input.

---

## 25. Teacher Forcing

At the beginning of training, KF updates can use noisy previous ground-truth boxes:

\[
\tilde{\mathbf{z}}_{k-1}
=
\mathbf{z}_{k-1}^{\mathrm{GT}}
+
\boldsymbol{\epsilon}
\]

Then gradually replace teacher-forced updates with detector predictions.

Recommended sequence:

1. noisy previous ground-truth boxes;
2. mixture of ground truth and predictions;
3. mostly predicted boxes;
4. fully online detector-generated updates.

This reduces the train–deployment mismatch.

---

## 26. Prior Corruption

Perfect priors create overdependence.

During training, randomly simulate:

- center-position errors;
- width and height errors;
- incorrect velocity;
- missed states;
- false states;
- stale states;
- covariance inflation;
- complete prior dropout.

For prior dropout:

\[
\mathbf{M}_k = \mathbf{0}
\]

for a random percentage of training windows.

The detector must remain functional when no valid KF prior exists.

---

## 27. Learnable KF Extension

Only after the fixed-KF version demonstrates an improvement should KF parameters be made learnable.

Keep fixed:

\[
\mathbf{F}(\Delta t),
\qquad
\mathbf{H}
\]

Potentially learn:

\[
\boldsymbol{\theta}_Q
=
[
\sigma_{a,x},
\sigma_{a,y},
\sigma_{a,w},
\sigma_{a,h}
]
\]

and:

\[
\boldsymbol{\theta}_R
=
[
\sigma_{c_x},
\sigma_{c_y},
\sigma_w,
\sigma_h
]
\]

Guarantee positive variances using:

\[
\sigma^2
=
\operatorname{softplus}(\rho)
+
\epsilon
\]

Do not initially learn:

- the full transition matrix;
- the measurement matrix;
- every covariance element;
- the Kalman gain;
- a neural replacement for the KF equations.

---

## 28. Optional Adaptive Measurement Noise

A later extension may predict the measurement covariance from:

\[
\mathbf{R}_k
=
r_{\phi}
\left(
s_k,
N_{\mathrm{events},k},
w_k,
h_k,
\tau_k
\right)
\]

where:

- \(s_k\): detector confidence;
- \(N_{\mathrm{events},k}\): local event count;
- \(w_k,h_k\): box size;
- \(\tau_k\): state age.

A very small MLP can output four positive variances.

This should remain compatible with real-time execution.

---

## 29. Real-Time Requirement

The complete runtime must satisfy:

\[
T_{\mathrm{repr}}
+
T_{\mathrm{KF}}
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
- \(T_{\mathrm{KF}}\): KF prediction and update;
- \(T_{\mathrm{prior}}\): prior-map generation;
- \(T_{\mathrm{network}}\): EventFPN–ConvGRU inference;
- \(T_{\mathrm{post}}\): decoding, NMS, association, and state management.

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

## 30. Computational Complexity

The KF operates on small \(8 \times 8\) matrices.

Its cost scales mainly with the number of active states:

\[
O(N_{\mathrm{tracks}})
\]

This is normally negligible compared with EventFPN and ConvGRU inference.

Avoid full-frame Gaussian generation in Python loops.

Generate each Gaussian only inside:

\[
\hat{c}_x \pm 3\sigma_x,
\qquad
\hat{c}_y \pm 3\sigma_y
\]

and use vectorized PyTorch or CUDA operations.

---

## 31. Real-Time Execution Pipeline

```text
Thread 1: DAVIS346 event acquisition
    └── Push timestamped events into a ring buffer

Thread 2: event representation construction
    └── Build the next event tensor

CPU tracking stage
    ├── Predict KF states
    ├── Associate detections
    ├── Update states
    └── Manage state creation and deletion

GPU stage
    ├── Generate or receive prior maps
    ├── Concatenate event and prior channels
    ├── Run EventFPN
    ├── Run ConvGRU
    └── Run the detection head
```

Use double buffering:

```text
Buffer A: currently processed
Buffer B: currently filled from DAVIS346
```

Event acquisition should continue while the previous window is being processed.

---

## 32. Operations to Avoid

Do not use:

- one KF update per raw event;
- event-to-track association for every event;
- one KF per pixel;
- Python loops over all image pixels;
- synchronous camera acquisition and inference;
- hard Kalman-guided cropping;
- full ConvGRU hidden-state warping;
- repeated CPU-to-GPU copies;
- a fully learned recurrent Kalman replacement in the first version.

Per-event association would approximately cost:

\[
O(N_{\mathrm{events}}N_{\mathrm{tracks}})
\]

and is not justified for the recommended design.

---

## 33. Evaluation Metrics

Because detection improvement is the main objective, prioritize:

- AP;
- AP\(_{50}\);
- AP\(_{75}\);
- precision;
- recall;
- false-positive rate;
- false-negative rate;
- localization error.

Evaluate difficult conditions separately:

- low event density;
- slow pedestrian motion;
- stationary pedestrians;
- rapid camera motion;
- partial occlusion;
- crowded scenes;
- new pedestrian entry.

---

## 34. Internal KF Diagnostics

Inspect:

- innovation residual;
- Mahalanobis distance;
- covariance growth;
- missed duration;
- prediction drift;
- association failures;
- state fragmentation;
- prior reliability.

Optional tracking metrics:

- HOTA;
- IDF1;
- ID switches;
- fragmentation.

These are diagnostic metrics, even when tracking is not the final objective.

---

## 35. Real-Time Metrics

Measure:

- event-representation time;
- KF prediction time;
- prior-map generation time;
- neural-network inference time;
- decoding and NMS time;
- association and state-update time;
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

## 36. Ablation Study

| Configuration | Event input | KF information | Learnable KF parameters |
|---|---|---|---|
| A | Original event tensor | None | No |
| B | Original event tensor | Occupancy map | No |
| C | Original event tensor | Occupancy + reliability + age | No |
| D | Original event tensor | Occupancy + reliability + age | Structured \(\mathbf{Q},\mathbf{R}\) |
| E | Original event tensor | Adaptive covariance priors | Small covariance network |
| F | Optional ego-motion-compensated events | Fixed KF priors | Optional |

The most important initial comparison is:

\[
\text{A versus C}
\]

Only when C clearly outperforms A should learnable covariance parameters be introduced.

---

## 37. Recommended Development Phases

### Phase 1 — Baseline profiling

Measure the original EventFPN–ConvGRU system:

- AP;
- recall;
- latency;
- memory;
- event-buffer behavior.

### Phase 2 — KF state bank

Implement:

- state initialization;
- prediction;
- association;
- measurement update;
- missed-state handling;
- state deletion.

Do not yet modify the detector.

### Phase 3 — Fixed prior maps

Add:

\[
M_{\mathrm{occ}},
\quad
M_{\mathrm{rel}},
\quad
M_{\mathrm{age}}
\]

Change the EventFPN input channels.

Retrain or fine-tune the network.

### Phase 4 — Robust sequential training

Add:

- teacher forcing;
- scheduled sampling;
- prior noise;
- false priors;
- missing priors;
- prior dropout.

### Phase 5 — KF calibration

Tune:

- \(\mathbf{Q}\);
- \(\mathbf{R}\);
- \(\mathbf{P}_0\);
- gating threshold;
- maximum missed duration;
- tentative-track confirmation;
- prior reliability thresholds.

### Phase 6 — Learnable covariance scalars

Make selected \(\mathbf{Q}\) and \(\mathbf{R}\) parameters learnable.

Compare against the fixed version.

### Phase 7 — Optional ego-motion EKF

For new DAVIS346 recordings with synchronized IMU data:

```text
DAVIS events + IMU
        ↓
Camera-motion EKF
        ↓
Ego-motion-compensated event stream
        ↓
Event representation
        ↓
EventFPN–ConvGRU with pedestrian KF priors
```

The camera-motion EKF and pedestrian bounding-box KF should remain separate modules.

---

## 38. Final Recommended Pipeline

\[
\boxed{
\begin{aligned}
&\text{DAVIS346 events}
\rightarrow
\text{event representation}
\\
&\text{previous detections}
\rightarrow
\text{per-pedestrian linear KF prediction}
\rightarrow
\text{soft prior maps}
\\
&[\text{event representation},\text{prior maps}]
\rightarrow
\text{EventFPN}
\rightarrow
\text{ConvGRU}
\rightarrow
\text{pedestrian detections}
\\
&\text{detections}
\rightarrow
\text{association}
\rightarrow
\text{KF update}
\rightarrow
\text{next inference cycle}
\end{aligned}
}
\]

---

## 39. Final Recommendation

The first implementation should use:

- a fixed linear KF;
- fixed \(\mathbf{F}\) and \(\mathbf{H}\);
- fixed and validated \(\mathbf{Q}\), \(\mathbf{R}\), and \(\mathbf{P}_0\);
- occupancy, reliability, and age prior maps;
- full-frame event input;
- vectorized prior generation;
- asynchronous acquisition and inference;
- no raw-event filtering;
- no hard cropping;
- no per-event KF updates;
- no learned Kalman gain;
- no fully learned KF replacement.

The KF must be present during training and deployment, but it should initially remain frozen.

The neural network learns how to use the KF-generated priors.

Only after the fixed architecture demonstrates a clear benefit should structured covariance parameters be made learnable.

---

## 40. Technical Summary

The most accurate statement of the method is:

\[
\boxed{
\text{The KF performs lightweight internal state tracking to enhance future pedestrian detection inference.}
}
\]

The recommended research sequence is:

\[
\boxed{
\text{Fixed KF priors}
\rightarrow
\text{validate detection improvement}
\rightarrow
\text{tune KF covariances}
\rightarrow
\text{optionally learn structured } \mathbf{Q},\mathbf{R}
}
\]

This design provides a practical balance among:

- mathematical validity;
- real-time feasibility;
- interpretability;
- numerical stability;
- compatibility with EventFPN and ConvGRU;
- controlled ablation;
- robustness to sparse or intermittent event observations.
