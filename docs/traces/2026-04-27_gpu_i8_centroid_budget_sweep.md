# 2026-04-27: GPU I8 Centroid-Budget Sweep

## Claim

The candidate-generation bottleneck should be attacked by reducing posting
fanout when recall allows it, before porting the current CPU loop to GPU.

Reason: the candidate-generation breakdown showed centroid selection, posting
accumulation, and final candidate top-k as the dominant non-full-window
substeps. `centroids_per_query_vector` directly controls selected centroids and
posting visits, so it is the smallest measurable policy lever.

## Change

Added a benchmark-only centroid-budget sweep:

- `python/kayak_bridge/gpu_i8_centroid_budget_sweep.py`
- `python/scripts/profile_gpu_i8_centroid_budget_sweep.py`
- `profile_gpu_i8_centroid_budget_sweep_raw`
- `profile_gpu_i8_centroid_budget_sweep`

The sweep varies `centroids_per_query_vector` while holding
`centroid_count`, `candidate_k`, document vectors, query vectors, and payload
fixed. For each row it reports:

- exact-reference candidate-window recall
- final i8 rerank recall
- CPU i8 candidate-generation timing
- CPU same-candidate i8 score timing
- candidate-generation breakdown
- selected centroid and posting-visit counts
- ratios against the baseline `centroids_per_query_vector=32`

Reason: this keeps vector counts, fanout, recall, and latency in one artifact,
which is required before changing the default policy.

## Measurement

Commands:

```bash
pixi run python -m py_compile python/kayak_bridge/gpu_i8_centroid_budget_sweep.py python/scripts/profile_gpu_i8_centroid_budget_sweep.py
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_rerank_contract.py python/tests/test_fastplaid_speed_track.py
pixi run python python/scripts/profile_gpu_i8_centroid_budget_sweep.py --case smoke:document_count=64,document_vector_count=8,query_count=1,query_vector_count=4,candidate_k=16 --centroid-budgets 4,8 --baseline-centroids-per-query-vector 8 --measurement-iterations 1 --output .cache/kayak/gpu_i8_centroid_budget_sweep/smoke.json
pixi run profile_gpu_i8_centroid_budget_sweep
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run profile_gpu_i8_centroid_budget_sweep_raw --case-set default --output .cache/kayak/gpu_i8_centroid_budget_sweep/default_summary.json
```

Artifacts:

- smoke report: `.cache/kayak/gpu_i8_centroid_budget_sweep/smoke.json`
- wide quiet log: `.cache/kayak/bench_quiet/20260427T092255Z`
- wide report: `.cache/kayak/gpu_i8_centroid_budget_sweep/summary.json`
- default quiet log: `.cache/kayak/bench_quiet/20260427T092425Z`
- default report:
  `.cache/kayak/gpu_i8_centroid_budget_sweep/default_summary.json`

## Results

Wide non-full case set status: `ok`, `3 / 3` cases.

| case | baseline cpqv | baseline recall | fastest no-loss cpqv | fastest no-loss recall | candidate generation s | candidate+score s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `32` | `0.65` | `4` | `0.70` | `0.0003779919995092011` | `0.0018763703325627528` |
| `doc_vectors64` | `32` | `0.65` | `16` | `0.65` | `0.00032525333275164786` | `0.0016399323327884` |
| `query_batch4` | `32` | `0.65` | `8` | `0.70` | `0.0005770263336065303` | `0.001573421667368772` |

Default non-full case set status: `ok`, `5 / 5` cases.

| case | baseline cpqv | baseline recall | fastest no-loss cpqv | fastest no-loss recall | candidate generation s | candidate+score s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `32` | `0.50` | `4` | `0.50` | `0.00012843299979673853` | `0.00042730033237603493` |
| `candidate32` | `32` | `0.15` | `4` | `0.15` | `0.00005996199979563244` | `0.00020366633240579782` |
| `query_vectors16` | `32` | `0.60` | `16` | `0.65` | `0.00022727066667963905` | `0.0007326199993258342` |
| `doc_vectors32` | `32` | `0.55` | `4` | `0.60` | `0.00013238666724646464` | `0.0005456643345193395` |
| `documents512` | `32` | `0.25` | `24` | `0.25` | `0.0002306100001684778` | `0.0005276539995975327` |

## Interpretation

Verified:

- the centroid-budget sweep reports timing, exact-reference recall, and posting
  fanout for non-full candidate windows
- lower centroid budgets preserve or improve the `32`-centroid baseline recall
  on all measured default and wide non-full synthetic cases
- the fastest no-loss budget is shape-dependent: measured winners include
  `4`, `8`, `16`, and `24`

Debunked:

- more selected centroids per query vector are not monotonically better for
  final i8 recall on these synthetic cases
- `centroids_per_query_vector=32` is not a justified fixed point for every GPU
  rerank candidate window

Not claimed:

- this does not prove a new public default
- this does not prove the same frontier on real encoded corpora
- this does not compare directly against FastPlaid full search; it is an
  internal policy sweep that should feed the next FastPlaid comparison

## Decision

Keep `32` as the benchmark baseline for comparability, but do not treat it as
the preferred optimization point.

Reason: the evidence supports a shape-aware budget policy or a calibration
step, not a single static replacement. The next implementation should evaluate
candidate-window recall and GPU top-k timing for selected budget policies, then
compare the best end-to-end Kayak row against FastPlaid CPU and CUDA on the same
explicit shapes.

## Validation

Observed:

- Python syntax check passed
- focused GPU/FastPlaid tests: `53 / 53` passed
- smoke budget sweep status: `ok`
- wide quiet budget sweep status: `ok`
- default quiet budget sweep status: `ok`
