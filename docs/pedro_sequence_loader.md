# PEDRo Sequence Loader

The sequence loader consumes:

```text
manifests/pedro_original_manifest.csv
```

and returns temporally ordered samples for event-memory training.

## Basic Usage

```python
import torch
from torch.utils.data import DataLoader

from src.event_memory import EventMemoryConfig, UncertaintyAwareEventMemory
from src.pedro_dataset import (
    EventRepresentationConfig,
    PedroManifestDataset,
    SequentialSequenceBatchSampler,
    pedro_collate,
)

rep_cfg = EventRepresentationConfig(
    representation="polarity_count",
    normalize="log1p",
)

dataset = PedroManifestDataset(
    "manifests/pedro_original_manifest.csv",
    split="train",
    representation_config=rep_cfg,
)

sampler = SequentialSequenceBatchSampler(dataset, batch_size=1)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=pedro_collate)

memory = UncertaintyAwareEventMemory(EventMemoryConfig(height=260, width=346))

for batch in loader:
    if batch["is_sequence_start"][0]:
        memory.reset(batch_size=1)

    event_tensor = batch["event_tensor"]
    priors = memory.make_priors(
        event_tensor,
        window_end_time=batch["window_end_time"][0],
    )
    model_input = torch.cat([event_tensor, priors], dim=1)
```

For the first causal memory experiment, use `batch_size=1`. Larger contiguous
batches are only correct if the training step unrolls memory inside the batch in
temporal order.

## Sample Fields

Each dataset sample contains:

```text
event_tensor          [C_event, 260, 346]
boxes                 [N, 4], xyxy pixel coordinates
labels                [N]
difficult             [N]
truncated             [N]
sequence_id
sequence_position
is_sequence_start
window_start_us
window_end_us
window_start_time     seconds
window_end_time       seconds
delta_t_s             None for sequence starts
npy_path
xml_path
```

## Event Representations

The loader currently supports:

```text
polarity_count: [2, H, W]
voxel_grid:     [2 * num_bins, H, W]
```

The first EventFPN layer must accept:

```text
C_event + 4
```

channels once memory priors are concatenated.
