# Prepared Exact Search Executor BrowseComp+ Gold Benchmark

## Claim

For repeated exact search against one already-published snapshot on the same
machine, the current `PreparedExactSearchExecutor` path:

- `PreparedExactSearchExecutor`

now reduces snapshot preparation cost materially relative to
`prepare_service_search_snapshot(...)` on this slice, because it can reuse a
directly materialized flat dim128 layout instead of fully loading nested packed
token vectors.

For steady-state search, the median quiet-wrapper result also favors the
executor path, but one of the three reruns inverted that ordering under heavy
host contention. The sound claim is therefore:

- executor preparation is clearly cheaper on this slice
- executor steady-state search is at least competitive with the direct prepared
  snapshot path, and the median of the current reruns is faster

This matters because it means the executor is no longer just a policy wrapper.
It owns a measurable materialization strategy that can cut repeated same-snapshot
startup cost without paying an obvious search-time penalty.

## Benchmark Setup

Benchmark entry point:

- [profile_prepared_exact_search_executor_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_prepared_exact_search_executor_browsecomp_gold.mojo:1)

Command:

```bash
pixi run mojo -I . benchmarks/profile_prepared_exact_search_executor_browsecomp_gold.mojo
```

Quiet-wrapper command used for decision-quality reruns:

```bash
bash scripts/run_bench_quiet.sh --repeats 3 --max-other-cpu 40 --timeout-seconds 10 --force -- pixi run mojo -I . benchmarks/profile_prepared_exact_search_executor_browsecomp_gold.mojo
```

Measured slice:

- dataset: `Tevatron/browsecomp-plus/gold-slice`
- slice: `browsecomp_plus_gold_slice`
- queries: `4`
- documents: `90`
- nominal query vectors: `32`
- nominal document vectors: `175`
- vector dim: `128`
- service root:
  `.cache/kayak/profile_prepared_exact_search_executor_browsecomp_gold`

The benchmark intentionally separates:

- snapshot preparation
- executor preparation
- direct prepared-snapshot search
- executor-backed search under different exact scorer configs

Reason:

- the question here was module overhead and same-snapshot repeated search cost,
  not transport overhead or end-to-end Python serving behavior
- before rerunning, the benchmark harness had to be updated to use
  `len(executor.prepared.segments)` because the executor now owns prepared
  segment state directly rather than exposing `prepared.snapshot.segments`

## Results

Quiet-wrapper median of 3 forced reruns:

- `prepare_snapshot`: `0.00707078431372549 s`
- `prepare_executor_default`: `0.0022043818181818184 s`
- `search_prepared_snapshot_direct`: `0.0010169268292682927 s`
- `search_executor_default`: `0.0006750802139037432 s`
- `search_executor_serial`: `0.0050233877551020415 s`
- `search_executor_parallel2`: `0.002410091603053435 s`

Derived comparisons:

- executor prepare vs prepared snapshot prepare:
  `3.207604x` faster, `68.824083%` lower mean time
- executor median search vs direct prepared-snapshot median search:
  `1.506379x` faster, `33.615655%` lower mean time
- forced serial vs default executor search:
  `7.441172x` slower
- forced `parallel_work_item_count_override = 2` vs default executor search:
  `3.570082x` slower

Per-rerun search medians for the two main paths:

- run 1:
  `search_prepared_snapshot_direct = 0.0010083414634146342 s`
  `search_executor_default = 0.0006598677248677249 s`
- run 2:
  `search_prepared_snapshot_direct = 0.0010169268292682927 s`
  `search_executor_default = 0.0006750802139037432 s`
- run 3:
  `search_prepared_snapshot_direct = 0.0013654051724137932 s`
  `search_executor_default = 0.0014764220183486239 s`

## Interpretation

What is verified:

- the prepared executor now has a materially cheaper prepare step on this slice
- that prepare win is large enough to survive repeated noisy reruns
- steady-state executor search is competitive with the direct prepared-snapshot
  path, and the current median favors the executor
- on this machine and slice, the default scorer configuration is better than
  forcing serial or forcing a two-work-item override

Why that conclusion is justified:

- the prepare delta is large across all three reruns, from roughly `2.68x` to
  `3.31x` faster for the executor path
- two of the three reruns favor executor search clearly, while one rerun
  reverses under extreme host contention
- using the median rather than the last run avoids letting that single noisy
  outlier dominate the conclusion
- the serial and forced-`parallel2` regressions were large enough to be far
  outside the observed run-to-run noise

## Uncertainty

This benchmark does not prove:

- the best Python API for concurrent same-snapshot serving
- the best process count versus intra-request parallelism tradeoff
- the best configuration on larger collections, different vector counts, or
  different hardware

The host was also not actually quiet during the quiet-wrapper reruns. The logs
show substantial external CPU activity from desktop applications and helper
processes, including one rerun with enough contention to invert the search-side
ordering. That is why the search claim above is phrased in terms of median
competitiveness rather than a universal per-run win. Tighter host isolation
would still be required for a publication-grade absolute latency claim.

## Bottom Line

The sound next conclusion is narrow:

- the repo now has a module-level exact-search executor seam that is explicit,
  measurable, and materially cheaper to prepare on this real slice
- the new direct materialization path does not show evidence of introducing a
  systematic steady-state search regression here

That is sufficient justification to keep the executor seam and build future
Python or serving concurrency work on top of it, rather than directly inside
transport code.
