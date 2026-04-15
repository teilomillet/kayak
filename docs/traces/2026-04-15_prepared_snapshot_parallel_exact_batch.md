# Prepared Snapshot Parallel Exact Batch

## Claim

The next question after the prepared snapshot and prepared executor work was:

- can one published snapshot support multiple exact queries concurrently inside
  Mojo without changing snapshot semantics
- what outer-request worker count is actually useful on one host
- how does outer concurrency interact with inner exact-scoring parallelism

The sound target was not "turn on threaded HTTP." It was:

- add one explicit module-level batch kernel for repeated exact search on a
  pinned prepared snapshot
- validate that it matches the serial prepared path exactly
- benchmark outer worker count against explicit exact-scoring configs before any
  Python transport decision

## Implementation

Added the new module:

- [prepared_exact_search_batch.mojo](/Users/teilomillet/Code/kayak/kayak/service/prepared_exact_search_batch.mojo:1)

This module owns:

- `PreparedExactSearchBatchConfig`
- `execute_search_batch_with_prepared_snapshot(...)`

Design choices and reasons:

- the batch kernel operates on
  [PreparedSearchSnapshot](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:51),
  not the higher-level executor object

Reason:

- the prepared snapshot is the real shared read-only resource
- the executor is still valuable for sequential reuse, but the concurrent
  resource we needed to prove was shared prepared-snapshot search

- each outer worker creates its own local `ExactCpuBackend` from the explicit
  `ExactScoringConfig`

Reason:

- this avoids assuming that one backend object should be shared across worker
  partitions
- it keeps scorer policy explicit at the batch boundary

- the kernel partitions request indices explicitly and writes one response slot
  per request

Reason:

- this makes outer-request parallelism measurable in isolation
- it avoids transport concerns, queueing policy, and mutation policy

Exported through:

- [kayak/service/__init__.mojo](/Users/teilomillet/Code/kayak/kayak/service/__init__.mojo:1)
- [kayak/__init__.mojo](/Users/teilomillet/Code/kayak/kayak/__init__.mojo:1)

Added focused correctness coverage:

- [test_service_prepared_exact_search_batch.mojo](/Users/teilomillet/Code/kayak/tests/test_service_prepared_exact_search_batch.mojo:1)

Added the benchmark surface:

- [profile_prepared_snapshot_parallel_exact_batch_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_prepared_snapshot_parallel_exact_batch_browsecomp_gold.mojo:1)

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_service_prepared_exact_search_batch.mojo
pixi run mojo -I . tests/test_service_prepared_snapshot_runtime.mojo
pixi run mojo -I . tests/test_service_prepared_exact_search_executor.mojo
pixi run mojo -I . benchmarks/profile_prepared_snapshot_parallel_exact_batch_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --repeats 3 --max-other-cpu 40 --timeout-seconds 10 --force -- \
  pixi run mojo -I . benchmarks/profile_prepared_snapshot_parallel_exact_batch_browsecomp_gold.mojo
```

Observed correctness results:

- `tests/test_service_prepared_exact_search_batch.mojo`: `2/2` passed
- `tests/test_service_prepared_snapshot_runtime.mojo`: `2/2` passed
- `tests/test_service_prepared_exact_search_executor.mojo`: `2/2` passed

What the new batch tests verify:

- shared prepared-snapshot batch execution matches the serial prepared path
  under the default exact scorer config
- the same is true under an explicit
  `parallel_work_item_count_override = 2`

The benchmark also performs a serial-versus-batch response equality check for
every measured `(worker_count, scoring_mode)` pair before timing.

## Benchmark Setup

Benchmark:

- [profile_prepared_snapshot_parallel_exact_batch_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_prepared_snapshot_parallel_exact_batch_browsecomp_gold.mojo:1)

Quiet-wrapper artifact directory:

- `.cache/kayak/bench_quiet/20260415T152343Z`

Wrapper metadata:

- `.cache/kayak/bench_quiet/20260415T152343Z/meta.txt`

Measured slice:

- dataset: `Tevatron/browsecomp-plus/gold-slice`
- slice: `browsecomp_plus_gold_slice`
- distinct queries: `4`
- request pool: `32`
- documents: `90`
- nominal query vectors: `32`
- nominal document vectors: `175`
- vector dim: `128`
- host-reported `parallelism_level`: `8`

Measured scoring modes:

- `default_auto`
- `serial_inner`
- `fixed2_inner`
- `shared_core_budget`

The shared-core-budget mode sets:

- `enable_parallel_work_item_oversubscription = False`
- `parallel_work_item_count_override = parallelism_level() // worker_count`

Reason:

- this is the simplest explicit way to test whether sharing the host between
  outer concurrent requests beats leaving the exact scorer fully automatic

## Results

Median of the 3 forced quiet-wrapper reruns:

- `w1 default_auto`: `0.032429909091 s` batch mean, `986.743 q/s`
- `w1 serial_inner`: `0.154771300000 s`, `206.757 q/s`
- `w1 fixed2_inner`: `0.080291900000 s`, `398.546 q/s`
- `w1 shared_core_budget`: `0.039412000000 s`, `811.935 q/s`
- `w2 default_auto`: `0.030398272727 s`, `1052.691 q/s`
- `w2 serial_inner`: `0.076899900000 s`, `416.125 q/s`
- `w2 fixed2_inner`: `0.043264727273 s`, `739.633 q/s`
- `w2 shared_core_budget`: `0.026639000000 s`, `1201.246 q/s`
- `w4 default_auto`: `0.021498000000 s`, `1488.511 q/s`
- `w4 serial_inner`: `0.038384090909 s`, `833.679 q/s`
- `w4 fixed2_inner`: `0.033516545455 s`, `954.752 q/s`
- `w4 shared_core_budget`: `0.035362300000 s`, `904.919 q/s`
- `w8 default_auto`: `0.021649333333 s`, `1478.106 q/s`
- `w8 serial_inner`: `0.021254916667 s`, `1505.534 q/s`
- `w8 fixed2_inner`: `0.022188181818 s`, `1442.209 q/s`
- `w8 shared_core_budget`: `0.021462500000 s`, `1490.973 q/s`

Best median row per worker count:

- `w1`: `default_auto`
- `w2`: `shared_core_budget`
- `w4`: `default_auto`
- `w8`: `serial_inner`

Top median cluster on this host:

- `w8 serial_inner`: `1505.534 q/s`
- `w8 shared_core_budget`: `1490.973 q/s`
- `w4 default_auto`: `1488.511 q/s`
- `w8 default_auto`: `1478.106 q/s`

Derived comparisons:

- best median row overall versus `w1 default_auto`:
  `1.525761x` higher throughput
- `w4 default_auto` versus `w1 default_auto`:
  `1.508509x` higher throughput
- spread between the best and worst row in the top cluster:
  `1.821812%`

## Stability

The current reruns do **not** justify naming one universal winner among the top
`w4`/`w8` candidates.

What is stable:

- outer-request concurrency clearly helps on this slice
- the best rows all use `worker_count >= 4`
- the top cluster is tightly packed, within `1.821812%`

What is not stable enough to overclaim:

- whether `w8 serial_inner` is meaningfully better than `w4 default_auto`
- whether a tuned inner policy is intrinsically better than the default once
  outer concurrency is already high

The host remained heavily contended during all three forced reruns, so these
medians are suitable for choosing the region worth keeping and exposing, but
not for hard-coding one exact serving default as universally optimal.

## Interpretation

What is verified:

- the repo now has an explicit module-level kernel for concurrent exact search
  batches on one pinned prepared snapshot
- the batch kernel preserves exact-search results relative to serial prepared
  execution
- outer-request concurrency can raise throughput meaningfully on the same
  snapshot, but not linearly

What the numbers suggest on this host:

- the sound operating region is `worker_count` between `4` and `8`
- `worker_count = 1` is clearly inferior for sustained `32`-request same-snapshot
  batches on this slice
- once outer concurrency is already high, multiple inner-scoring policies become
  competitive; the host noise is too high to justify treating one as a global
  winner
- the measured value is therefore the explicit batch kernel and the policy
  surface, not a one-line hardcoded scheduler rule

What this does **not** prove:

- that the hosted Python HTTP server is now concurrent
- that one in-process Python API should expose this batch kernel directly
- that the same worker/scoring choice will remain best on larger collections or
  on different hardware

## Hosted Python Status

The current hosted transport is still single-process and single-threaded:

- [server.py](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:7)
  imports `HTTPServer`
- [KayakEngineHttpServer](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:28)
  subclasses `HTTPServer`
- [do_POST(...)](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:71)
  dispatches one request synchronously

So the sound conclusion is:

- the Mojo module layer now has a verified same-snapshot multi-query kernel
- the hosted Python edge still does not expose concurrent search on top of it

## Bottom Line

The repo now supports multiple exact queries against the same prepared snapshot
at the Mojo module layer, with correctness checks and a measured worker/scoring
matrix.

The best current evidence on this host is:

- use the prepared-snapshot batch kernel for module-level concurrency
- treat `worker_count` in the `4` to `8` range as the right search space for
  exact same-snapshot throughput on this slice
- avoid hard-coding one specific `(worker_count, scoring_mode)` winner until
  the same matrix is rerun on a quieter host or a larger serving slice
