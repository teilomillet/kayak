# 2026-04-15: Python Prepared Exact Runtime Lane Sweep

## Claim

After adding `concurrency_lane_count` to the Python prepared exact-search
runtime, the next question was:

- how much same-snapshot throughput does the Python runtime gain from multiple
  execution lanes
- how does lane count interact with the inner Mojo batch-kernel `worker_count`
- does an explicit shared-host inner-work budget help more than the default
  exact scorer policy on this host

This trace measures the Python runtime surface directly rather than inferring
from the lower-level Mojo batch kernel alone.

## Implementation

Added the benchmark script:

- `python/scripts/bench_prepared_exact_runtime.py`

What it does:

- loads one encoded Python task JSON
- stages that task into a temporary hosted-engine service root
- prepares one pinned exact-search session for correctness reference
- sweeps:
  - `concurrency_lane_count`
  - `worker_count`
  - scoring mode
- validates every runtime configuration against the prepared session before
  timing
- measures one full request-pool batch through the Python runtime surface
- prints one structured `RESULT` line per configuration

Why this script was justified:

- the new runtime lane contract needed real evidence at the Python boundary
- the existing Mojo benchmark only answered the inner prepared-snapshot batch
  kernel question
- users care about the actual Python runtime surface, not only the lower-level
  Mojo primitive

## Measured Setup

Command:

```bash
bash scripts/run_bench_quiet.sh --repeats 3 --max-other-cpu 40 --timeout-seconds 30 --force -- \
  pixi run python python/scripts/bench_prepared_exact_runtime.py \
    --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json \
    --lane-counts 1,2,4 \
    --worker-counts 1,2,4 \
    --scoring-modes default_auto,shared_host_budget \
    --warmup-iterations 2 \
    --measurement-iterations 8 \
    --request-pool-count 32 \
    --max-batch-size 32 \
    --max-batch-wait-ms 5
```

Quiet-wrapper log directory:

- `.cache/kayak/bench_quiet/20260415T162604Z`

Task:

- dataset: `Tevatron/browsecomp-plus/gold-slice`
- slice: `browsecomp_plus_gold_slice`
- documents: `90`
- distinct queries: `4`
- request pool: `32`
- vector dim: `128`

Scoring modes:

- `default_auto`
- `shared_host_budget`

`shared_host_budget` sets:

- `enable_parallel_work_item_oversubscription = False`
- `parallel_work_item_count_override = host_cpu_count // (lane_count * worker_count)`

## Validation

Every measured configuration first validated:

- Python prepared runtime results matched the pinned prepared-session results

The benchmark command completed successfully under all three repeated outer
runs.

## Important Uncertainty

All three quiet-wrapper runs timed out waiting for a quiet host and then
proceeded under `--force`.

Observed competing load included:

- persistent editor and terminal activity
- repeated `modular-crashpad-handler` activity
- during run 3, a separate Mojo benchmark:
  `mojo -I . benchmarks/legal_rag_bench_real_subset_search.mojo`

So these numbers are useful for region finding and debunking bad settings, but
they are not clean enough to justify a universal default.

## Median Results

Median across the 3 outer quiet-wrapper runs:

| lanes | workers | mode | median batch s | median q/s | median executed batches | median avg batch size |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | `default_auto` | `0.047354911367` | `675.748` | `8` | `32.000` |
| 1 | 1 | `shared_host_budget` | `0.040657515478` | `787.062` | `8` | `32.000` |
| 1 | 2 | `default_auto` | `0.037443599009` | `854.619` | `8` | `32.000` |
| 1 | 2 | `shared_host_budget` | `0.040025536233` | `799.490` | `8` | `32.000` |
| 1 | 4 | `default_auto` | `0.052348182260` | `611.292` | `9` | `28.444` |
| 1 | 4 | `shared_host_budget` | `0.040285265524` | `794.335` | `8` | `32.000` |
| 2 | 1 | `default_auto` | `0.045039640740` | `710.485` | `16` | `15.067` |
| 2 | 1 | `shared_host_budget` | `0.042923489644` | `745.513` | `17` | `15.059` |
| 2 | 2 | `default_auto` | `0.038932703232` | `821.931` | `16` | `16.000` |
| 2 | 2 | `shared_host_budget` | `0.045559062622` | `702.385` | `16` | `16.000` |
| 2 | 4 | `default_auto` | `0.041346968632` | `773.938` | `16` | `16.000` |
| 2 | 4 | `shared_host_budget` | `0.050207994733` | `637.349` | `16` | `16.200` |
| 4 | 1 | `default_auto` | `0.043974958360` | `727.687` | `32` | `8.000` |
| 4 | 1 | `shared_host_budget` | `0.052172369731` | `613.351` | `32` | `7.758` |
| 4 | 2 | `default_auto` | `0.038459624993` | `832.041` | `32` | `8.000` |
| 4 | 2 | `shared_host_budget` | `0.062129437254` | `515.054` | `32` | `8.000` |
| 4 | 4 | `default_auto` | `0.035854833244` | `892.488` | `32` | `8.000` |
| 4 | 4 | `shared_host_budget` | `0.043942338496` | `728.227` | `32` | `8.000` |

## Top Cluster

Top median configurations on this host:

- `lanes=4, workers=4, default_auto`: `892.488 q/s`
- `lanes=1, workers=2, default_auto`: `854.619 q/s`
- `lanes=4, workers=2, default_auto`: `832.041 q/s`
- `lanes=2, workers=2, default_auto`: `821.931 q/s`

Baseline:

- `lanes=1, workers=1, default_auto`: `675.748 q/s`

Derived comparisons:

- `4x4 default_auto` vs baseline: `1.321x`
- `1x2 default_auto` vs baseline: `1.265x`
- `4x2 default_auto` vs baseline: `1.231x`

## Interpretation

What is clearly true:

- multiple runtime lanes can help at the Python surface
- raising `worker_count` from `1` to `2` is already a strong move on this host
- the Python runtime does not need a new mechanism before benchmarking; the
  current multi-lane process backend already reaches a meaningfully better
  throughput region

What the current measurements suggest:

- `default_auto` generally beats `shared_host_budget` once either
  `lane_count > 1` or `worker_count > 1`
- the useful search region on this host is roughly:
  - `worker_count` in `2..4`
  - `concurrency_lane_count` in `1..4`
- `4x4 default_auto` is the best median point in this sweep, but not by enough
  margin to justify calling it a universal winner under these host conditions

One important instability:

- `lanes=1, workers=4, default_auto` was highly unstable
- its median result was depressed by underfilled batches and noisy-host runs
- `shared_host_budget` stabilized that configuration by restoring full-size
  batches, but still did not beat the best `default_auto` points

## What This Does Not Prove

- that `4x4 default_auto` should become the global default
- that the same lane/worker region will hold on larger collections or different
  hardware
- that the runtime memory cost of extra lanes is always worth the gain

## Guidance

For this current host and workload slice:

- start the search around `worker_count=2` before increasing inner parallelism
  further
- treat `concurrency_lane_count=2..4` as the region worth testing when multiple
  same-snapshot callers matter
- avoid assuming that a manual shared-host budget beats the default exact
  scorer policy once you already have multiple lanes or multiple batch workers

The next evidence-backed follow-on is:

- add resident-memory measurement to the same sweep, so the throughput gain can
  be evaluated against the extra prepared-snapshot memory duplicated per lane
