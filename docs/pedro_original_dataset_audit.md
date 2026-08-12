# PEDRo Original Dataset Audit

Dataset path:

```text
/media/birb/grvc/PEDRo_original
```

Audit environment:

```text
/home/birb/Documents/Event_based_datasets/yolo_venv
```

## Verdict

The dataset is suitable for the event-plus-prior investigation, with caveats.

It provides the essential requirements:

- raw event windows with timestamps;
- DAVIS346 spatial resolution, `346 x 260`;
- Pascal VOC-style pedestrian boxes;
- train/val/test splits;
- additional no-person windows for negative-memory and false-positive testing.

It is not suitable for a naive randomly shuffled frame dataset. The memory
experiment needs a sequence manifest that preserves temporal order, computes
`window_end_time` from the event timestamps, and resets memory at inferred
recording discontinuities.

## Directory Layout

Main person dataset:

| Split | Event files | XML files |
| --- | ---: | ---: |
| `numpy/train`, `xml/train` | 19,228 | 19,228 |
| `numpy/val`, `xml/val` | 3,950 | 3,950 |
| `numpy/test`, `xml/test` | 3,823 | 3,823 |

Extra person-only live subset:

| Split | Event files | XML files |
| --- | ---: | ---: |
| `live_annotated_dataset_person_only/numpy/train` | 1,459 | 1,459 |

Negative subsets:

| Split | Event files | XML files | Objects |
| --- | ---: | ---: | ---: |
| `valid_set_no_person/numpy/val` | 700 | 700 | 0 |
| `NO_PERSPNS/PEDRo_original/with_no_persons/numpy/train` | 2,914 | 2,914 | 0 |

The dataset also contains preview PNGs. Those are useful for inspection but are
not required for training.

## Event Format

Each loadable `.npy` file is shaped:

```text
[N, 4]
```

The columns are:

```text
[timestamp, x, y, polarity]
```

Observed ranges across loadable files:

| Column | Min | Max |
| --- | ---: | ---: |
| `x` | 0 | 345 |
| `y` | 0 | 259 |
| `polarity` | 0 | 1 |

Timestamps are absolute microsecond-scale integers. Use:

```python
window_start_time = events[0, 0]
window_end_time = events[-1, 0]
delta_t = (window_end_time_k - window_end_time_k_minus_1) / 1_000_000
```

Memory propagation must use these event timestamps, not dataloader or inference
wall-clock time.

## Window Timing

Most main-set windows are approximately 40 ms.

| Split | Duration median | Duration notable range |
| --- | ---: | ---: |
| main train | 40,000 us | 14,070 to 426,830 us |
| main val | 40,000 us | 36,375 to 230,368 us |
| main test | 40,000 us | 11,160 to 1,126,192 us |
| live person-only train | 39,994 us | 39,766 to 40,000 us |
| no-person val | 40,000 us | fixed 40,000 us |
| no-person train | 40,000 us | fixed 40,000 us |

Some consecutive files have gaps, skipped intervals, negative jumps, or very
large recording jumps. The flat filename order is still useful, but it must be
converted into explicit sequences.

Recommended initial reset rule:

- reset on split/subset boundary;
- reset when the next start timestamp is earlier than the previous end timestamp;
- reset when the gap is clearly a recording break, initially `gap > 1.0 s`;
- keep shorter gaps and propagate memory using the actual timestamp delta.

This preserves memory through skipped windows while avoiding carryover across
unrelated recordings.

## Annotation Format

Annotations are Pascal VOC-style XML files with:

```text
width = 346
height = 260
class = person
```

Main annotation counts:

| Split | Person boxes | Empty XMLs |
| --- | ---: | ---: |
| main train | 34,708 | 0 |
| main val | 4,372 | 0 |
| main test | 4,179 | 1 |

Main train object counts per window:

| Objects | Windows |
| ---: | ---: |
| 1 | 10,475 |
| 2 | 5,457 |
| 3 | 608 |
| 4 | 1,945 |
| 5 | 743 |

The main set has almost no negative examples. Use the negative subsets during
training/evaluation so the detector learns to reject empty scenes and stale
false memory.

## Data Integrity Issues

Two main train files failed memory-mapped loading:

```text
/media/birb/grvc/PEDRo_original/numpy/train/frame0010206.npy
/media/birb/grvc/PEDRo_original/numpy/train/frame0010207.npy
```

Exclude or repair these before training.

All parsed XML boxes were within the `346 x 260` image bounds.

## Temporal Label Coherence

Adjacent annotations are generally coherent enough for memory training.

Median mean best-IoU between adjacent windows:

| Split | Gap <= 1s | Gap <= 100ms |
| --- | ---: | ---: |
| main train | 0.897 | 0.898 |
| main val | 0.915 | 0.917 |
| main test | 0.894 | 0.901 |
| live person-only train | 0.919 | 0.919 |

This supports detection-informed memory updates and teacher-forced/scheduled
sampling updates. There are still object-count changes and low-IoU transitions,
so memory corruption/dropout remains important.

## Suitability For Priors

Suitable:

- belief priors can be updated from previous predictions or teacher-forced boxes;
- uncertainty and age can be propagated with event timestamp deltas;
- support maps can be generated directly from each event window;
- no-person subsets allow false-memory persistence tests;
- multiple-pedestrian windows exist in the main train split.

Caveats:

- no explicit sequence IDs are stored in the directory layout;
- main splits are flattened and contain discontinuities;
- two train event files are bad;
- the main train/val folders are strongly positive-only;
- batches must not shuffle windows within a sequence for memory training;
- negative subsets have different timestamp ranges and should be treated as
  separate sequences/subsets.

## Required Preprocessing Before Training

Create a manifest with one row per loadable event window:

```text
subset
split
sequence_id
window_index
npy_path
xml_path
window_start_us
window_end_us
duration_us
gap_from_previous_us
is_sequence_start
num_boxes
```

Then train with a sequential sampler or sequence-level batches:

1. reset memory at `is_sequence_start`;
2. load events and build the event representation;
3. call `memory.make_priors(..., window_end_time=window_end_us / 1e6)`;
4. concatenate the four prior channels with the event tensor;
5. run the detector;
6. update memory after inference.
