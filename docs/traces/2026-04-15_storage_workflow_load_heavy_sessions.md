# Storage Workflow Trace: Load-Heavy Sessions

## Objective

The prior storage trace verified that `binary_f16_le` halves artifact
size but tends to regress load time.

That still left an open question:

- if a workload loads a stored index and then serves multiple exact
  searches from that loaded artifact
- is there any realistic session size where `binary_f16_le` becomes the
  better default

This trace answers that directly with repeated `load + N searches`
sessions.

## Benchmark Shape

The new benchmark excludes build time on purpose.

Reason:

- the default encoding decision for persisted artifacts is about serving
  from already-built storage
- including build would blur the startup question with an offline ingest
  concern

One benchmark session is:

1. `load_stored_packed_index(root)`
2. run `queries_per_load` exact searches
3. report total session wall time

Queries cycle through the task’s judged queries, so `queries_per_load`
may exceed the number of unique queries in the task.

## New Surfaces

- `benchmarks/task_json_storage_workflow_compare.mojo`
- `python/kayak_bridge/task_storage_workflow.py`
- `python/scripts/bench_task_storage_workflow.py`
- `python/tests/test_task_storage_workflow.py`

## Commands

Standard query-count ladder:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_workflow.py \
  --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json

env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_workflow.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json

env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_workflow.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json
```

Heavier probe on the best candidate slice:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_workflow.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-prefix r2med_biology_real_subset_storage_workflow_heavy \
  --queries-per-load 32 --queries-per-load 64 --queries-per-load 128

env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_workflow.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-prefix r2med_biology_real_subset_storage_workflow_256 \
  --queries-per-load 256
```

## Results

Artifacts:

- BRIGHT: `.cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_storage_workflow_bundle.json`
- LEMB: `.cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_storage_workflow_bundle.json`
- R2MED: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_workflow_bundle.json`
- R2MED heavy: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_workflow_heavy_storage_workflow_bundle.json`
- R2MED 256: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_workflow_256_storage_workflow_bundle.json`

Ratios are total session time `binary_f16_le / binary_le`.

### BRIGHT

| Queries Per Load | Session Ratio |
| --- | ---: |
| `1` | `1.1286` |
| `2` | `1.1300` |
| `4` | `1.1272` |
| `8` | `1.0784` |
| `16` | `1.0612` |
| `32` | `1.1650` |

No observed crossover.

### LEMB

| Queries Per Load | Session Ratio |
| --- | ---: |
| `1` | `1.1706` |
| `2` | `1.1232` |
| `4` | `1.0998` |
| `8` | `1.1043` |
| `16` | `1.0856` |
| `32` | `1.0560` |

No observed crossover.

### R2MED

| Queries Per Load | Session Ratio |
| --- | ---: |
| `1` | `1.1500` |
| `2` | `1.1333` |
| `4` | `1.1262` |
| `8` | `1.0918` |
| `16` | `1.0636` |
| `32` | `1.0245` |

R2MED was the closest measured case, so it was extended:

| Queries Per Load | Session Ratio |
| --- | ---: |
| `32` | `1.0381` |
| `64` | `1.0276` |
| `128` | `1.0024` |
| `256` | `1.0090` |

Still no observed crossover.

## Interpretation

Measured fact:

- `binary_f16_le` becomes less bad as more searches amortize load cost
- but on these measured slices it never becomes faster overall
- even on the best candidate slice, no total-session crossover was
  observed through `256` searches per load

The most important implication is negative:

- the earlier decision stands
- there is still not enough evidence to change the default packed-index
  encoding from `binary_le` to `binary_f16_le`

`binary_f16_le` remains a strong explicit option when artifact size is
the main objective, especially because it doubles Kayak’s storage edge
over LanceDB in the measured storage-scale comparisons.

But the workflow benchmark shows that storage compression alone does not
translate into a better default serve path.

## Residual Uncertainty

This benchmark measures repeated warm-host sessions, not fully cold-cache
machine restarts.

That means one uncertainty remains:

- a much colder I/O regime could shift the crossover point further toward
  `binary_f16_le`

But that would require a different benchmark setup with explicit cache
control. Within the current local workflow, no crossover was observed.
