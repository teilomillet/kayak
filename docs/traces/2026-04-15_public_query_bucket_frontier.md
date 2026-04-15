# 2026-04-15: modular public query-bucket frontier

## Claim

If we want to avoid tuning Kayak to one query shape, we need a benchmark layer
that can slice existing judged tasks by actual query vector count without
changing search, planner, or storage behavior.

The intended claim is narrow:

- add benchmark-side query-vector buckets
- keep vector count explicit in the output
- reuse existing stage-aware measurement code
- do not refactor planner or retrieval kernels

## Why this was justified

I checked the local benchmark code before editing:

- datasets were already modular in
  [kayak/benchmarks/public_benchmark_dataset.mojo](../../kayak/benchmarks/public_benchmark_dataset.mojo)
- the public runners were still script-specific, for example
  [benchmarks/public_small_window_frontier.mojo](../../benchmarks/public_small_window_frontier.mojo)
- there was no reusable layer for slicing a judged task by actual query vector
  count

That meant the smallest justified addition was benchmark orchestration, not a
planner or kernel change.

## What changed

New benchmark-side query bucketing primitive:

- [kayak/benchmarks/query_vector_bucket.mojo](../../kayak/benchmarks/query_vector_bucket.mojo)

What it owns:

- machine-readable query-vector bucket metadata
- standard power-of-two-style bucket construction
- query-only task slicing for benchmark use

Why the task slice keeps `documents` empty:

- stage-aware frontier measurement searches against the mirrored snapshot, not
  `task.documents`
- copying the full encoded document list once per bucket would create
  avoidable O(bucket_count) memory and copy cost
- the helper is named explicitly as a query-slicing benchmark helper so that
  this tradeoff stays visible

New JSON wrapper:

- [kayak/benchmarks/query_bucket_stage_aware_json.mojo](../../kayak/benchmarks/query_bucket_stage_aware_json.mojo)

Reason:

- keep bucket metadata explicit without mutating the existing
  `StageAwareSearchSummary` contract

New public runner:

- [benchmarks/public_query_bucket_frontier.mojo](../../benchmarks/public_query_bucket_frontier.mojo)

Reason:

- reuse the exact same plan family as the small-window public frontier
- only add query-bucket slicing around it

Task wiring:

- [pyproject.toml](../../pyproject.toml)

Test coverage:

- [tests/test_query_vector_bucket.mojo](../../tests/test_query_vector_bucket.mojo)

## Validation

Targeted tests:

- `pixi run bash -lc 'mojo -I . tests/test_query_vector_bucket.mojo'`
  - `4/4` passed

Full benchmark runner:

- `pixi run bash -lc 'mojo -I . benchmarks/public_query_bucket_frontier.mojo'`
  - completed successfully
  - wrote `.cache/kayak/public_query_bucket_frontier.json`

## Measured outcome

I inspected the produced artifact directly.

Verified shape:

- `75` rows total
- top-level fields:
  - `query_vector_bucket_key`
  - `query_vector_bucket_label`
  - `query_vector_bucket_min`
  - `query_vector_bucket_max`
  - `measured`

The important empirical result was:

- `beir/scifact/test -> ['qv_17_32']`
- `beir/fiqa/test -> ['qv_17_32']`
- `orionweller/LIMIT-small -> ['qv_17_32']`
- `Tevatron/browsecomp-plus/evidence-slice -> ['qv_17_32']`
- `Tevatron/browsecomp-plus/gold-slice -> ['qv_17_32']`

Interpretation:

- the new modular runner works
- the current public suite does **not** currently span multiple query-vector
  regimes
- so optimizing only on these slices would still risk overfitting to a single
  query-shape band

## Decision

Keep this addition.

Reason:

- it is additive rather than architectural
- it passed the local benchmark-side tests
- it produced a concrete new fact about the current benchmark suite:
  current public slices are all clustered in the same `17-32` query-vector
  bucket

That result is directly actionable:

- keep the query-bucket runner
- add new benchmark families such as MIRACL and BRIGHT to widen the measured
  query-shape distribution
- avoid treating current public slices as sufficient coverage for planner or
  stage-1 tuning by query complexity
