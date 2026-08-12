# Event-Native Detector Memory Integration

Local detector-memory config:

```text
configs/event_native_memory_pedro_p2_centerness_lite96_crop224.yaml
```

This config is derived from the previous event-native detector experiment, but
the runtime code used by this repository is now vendored locally under
`models/`, `datasets/`, `training/`, and `small_object_approach/`.

The detector currently expects three temporal stage tensors:

```text
dts_ms = [10, 20, 40]
stage tensor shape = [B, 11, 224, 224]
```

Those 11 channels come from:

```text
4 time bins * 2 polarities = 8
2 recency channels
1 count channel
```

## Memory-Prior Adaptation

For the first memory experiment, keep the detector architecture intact except for
the first input convolution:

```text
old input channels = 11
new input channels = 11 + 4 = 15
```

Append the same four memory-prior channels to every temporal stage:

```python
priors = memory.make_priors(anchor_stage, window_end_time=t_end)
stage_tensors = append_priors_to_stage_tensors(stage_tensors, priors, image_size=224)
```

The memory is kept in the detector's `224 x 224` square coordinate system for
this integration, because the detector targets, losses, decoded boxes, and
training crops all operate in that coordinate system.

## Required Training Change

Do not use the old training sampler unchanged. It uses shuffled or weighted
frame sampling, which destroys causal memory.

Use:

```text
ManifestEventNativeSequenceDataset
SequentialEventNativeBatchSampler(batch_size=1)
collate_event_native_memory
```

Then reset memory when:

```python
if meta["is_sequence_start"]:
    memory.reset(batch_size=1)
```

## Checkpoint Reuse

The existing checkpoint can initialize the 15-channel model. The original
11-channel first convolution is copied into the first 11 channels, and the four
new prior channels are zero-initialized.

Supported by:

```text
load_checkpoint_with_expanded_input
```

## Smoke Test

Run:

```bash
source /home/birb/Documents/Event_based_datasets/yolo_venv/bin/activate
python scripts/smoke_event_native_memory_detector.py --device cpu --steps 2
```

This verifies:

- manifest-backed sequential loading;
- exact old detector stage tensor construction;
- prior concatenation to 15 channels;
- checkpoint loading with expanded input convolution;
- forward pass;
- dense detection loss;
- backward pass;
- teacher-forced memory update after inference.

To initialize from the previous event-only checkpoint, pass it explicitly:

```bash
python scripts/train_event_native_memory.py \
  --init-checkpoint /path/to/event_only_checkpoint.pth
```
