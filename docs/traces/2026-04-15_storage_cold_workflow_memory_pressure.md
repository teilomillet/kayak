# Storage Cold Workflow Trace: `memory_pressure`

## Objective

The warm workflow trace answered one question:

- after a stored packed index is already in the local host’s working set,
  does `binary_f16_le` ever overtake `binary_le` as more searches amortize
  load cost

This trace answers the colder version:

- after an explicit cache-perturbation step
- does `binary_f16_le` overtake `binary_le` for a load-heavy exact-search
  session

## Constraint

True cache flush was not available in this environment.

Verified locally:

- `purge` exists on this macOS host
- `purge` returns `Unable to purge disk buffers: Operation not permitted`

So the benchmark uses an explicit cold-ish policy instead of claiming a
true cold-cache guarantee.

## Cold Policy

Policy name:

- `memory_pressure_percent_free`

Parameters used in the measured runs:

- `percent_free = 60`
- `sample_seconds = 1`
- `hysteresis_seconds = 1`

Command shape:

```bash
memory_pressure -Q -p 60 -s 1 -y 1
```

Why this policy was chosen:

- it is available without elevated privileges
- it measurably perturbs VM state on this host
- stronger settings such as `-p 5` or `-p 20` were too expensive to use
  repeatedly inside a session benchmark

Observed validation signal before wiring the benchmark:

- file-backed pages dropped materially after the probe
- the first subsequent load-heavy session moved away from the warm-path
  numbers

This is still only a cold-ish boundary, not a hard page-cache flush.

## Benchmark Shape

The cold harness splits artifact preparation from measurement.

Reason:

- preparing or rewriting the artifact immediately before timing would
  warm the very pages we want to treat as colder startup input

Measured session:

1. apply `memory_pressure`
2. `load_stored_packed_index(root)`
3. run `queries_per_load` exact searches
4. report one total session time

Each point is the mean of `3` cold repetitions.

## New Surfaces

- `benchmarks/task_json_storage_artifact_prepare.mojo`
- `benchmarks/task_json_storage_single_session.mojo`
- `python/kayak_bridge/task_storage_cold_workflow.py`
- `python/scripts/bench_task_storage_cold_workflow.py`
- `python/tests/test_task_storage_cold_workflow.py`

## Commands

Representative cold matrix:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_cold_workflow.py \
  --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json \
  --repetitions 3 --cold-percent-free 60

env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_cold_workflow.py \
  --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json \
  --repetitions 3 --cold-percent-free 60

env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_cold_workflow.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --repetitions 3 --cold-percent-free 60
```

Heavier candidate check:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 500 -- \
  uv run --python 3.11 python python/scripts/bench_task_storage_cold_workflow.py \
  --task .cache/kayak/r2med_biology_real_subset/python_task.json \
  --artifact-prefix r2med_biology_real_subset_storage_cold_workflow_128 \
  --queries-per-load 128 \
  --repetitions 3 --cold-percent-free 60
```

## Results

Artifacts:

- BRIGHT: `.cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_storage_cold_workflow_bundle.json`
- LEMB: `.cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_storage_cold_workflow_bundle.json`
- R2MED: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_cold_workflow_bundle.json`
- R2MED 128: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_cold_workflow_128_storage_cold_workflow_bundle.json`

Ratios are total session time `binary_f16_le / binary_le`.

### BRIGHT

| Queries Per Load | Binary Session s | F16 Session s | Ratio |
| --- | ---: | ---: | ---: |
| `1` | `0.027105` | `0.030500` | `1.1253` |
| `8` | `0.039039` | `0.039798` | `1.0195` |
| `32` | `0.069799` | `0.072821` | `1.0433` |

No observed crossover.

### LEMB

| Queries Per Load | Binary Session s | F16 Session s | Ratio |
| --- | ---: | ---: | ---: |
| `1` | `0.073983` | `0.081911` | `1.1072` |
| `8` | `0.116133` | `0.117604` | `1.0127` |
| `32` | `0.230720` | `0.256080` | `1.1099` |

No observed crossover.

### R2MED

| Queries Per Load | Binary Session s | F16 Session s | Ratio |
| --- | ---: | ---: | ---: |
| `1` | `0.028686` | `0.030988` | `1.0803` |
| `8` | `0.039226` | `0.042584` | `1.0856` |
| `32` | `0.079974` | `0.084079` | `1.0513` |
| `128` | `0.229202` | `0.236769` | `1.0330` |

No observed crossover.

## Warm vs Cold Comparison

The colder policy changes the exact ratios, but not the decision.

Representative comparison:

| Dataset | Queries Per Load | Warm Ratio | Cold-ish Ratio |
| --- | ---: | ---: | ---: |
| BRIGHT | `1` | `1.1286` | `1.1253` |
| BRIGHT | `8` | `1.0784` | `1.0195` |
| BRIGHT | `32` | `1.1650` | `1.0433` |
| LEMB | `1` | `1.1706` | `1.1072` |
| LEMB | `8` | `1.1043` | `1.0127` |
| LEMB | `32` | `1.0560` | `1.1099` |
| R2MED | `1` | `1.1500` | `1.0803` |
| R2MED | `8` | `1.0918` | `1.0856` |
| R2MED | `32` | `1.0245` | `1.0513` |

The direction is not perfectly monotonic across tasks, but the important
fact is stable:

- no measured cold-ish point flipped in favor of `binary_f16_le`

## Decision Impact

This closes the main remaining uncertainty from the warm trace.

Measured conclusion:

- under the chosen cold-ish startup policy
- across BRIGHT, LEMB, and R2MED
- and through a heavier `128`-query session on the closest candidate
  slice
- `binary_f16_le` still did not beat `binary_le`

So the default packed-index encoding should remain `binary_le`.

`binary_f16_le` is still justified as an explicit user-selectable option
when storage footprint matters more than startup latency.

## Residual Uncertainty

One uncertainty still remains:

- a true cold page-cache flush could move the numbers further

That was not measurable here because `purge` was not permitted, so this
trace should be read as:

- stronger than the warm-host benchmark
- weaker than a privileged true cold-cache benchmark
