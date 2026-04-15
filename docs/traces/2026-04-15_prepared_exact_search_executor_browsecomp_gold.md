# Prepared Exact Search Executor BrowseComp+ Gold Benchmark

## Claim

For repeated exact search against one already-published snapshot on the same
machine, the new module seam:

- `PreparedExactSearchExecutor`

does not add measurable steady-state search overhead relative to calling
`execute_search_with_prepared_snapshot(...)` directly on the same prepared
snapshot.

This matters because it means the new executor is a real reusable kernel, not a
policy wrapper that adds latency by itself.

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

## Results

Quiet-wrapper mean of 3 reruns:

- `prepare_snapshot`: `0.006857503144654 s`
- `prepare_executor_default`: `0.006862742138365 s`
- `search_prepared_snapshot_direct`: `0.001067950456323 s`
- `search_executor_default`: `0.001071412429379 s`
- `search_executor_serial`: `0.004749038548753 s`
- `search_executor_parallel2`: `0.002429537467700 s`

Derived comparisons:

- executor search overhead vs direct prepared-snapshot search:
  `+0.324170%`
- forced serial vs default executor search:
  `4.432503x` slower
- forced `parallel_work_item_count_override = 2` vs default executor search:
  `2.267602x` slower

## Interpretation

What is verified:

- the prepared executor wrapper is effectively free on this slice for
  steady-state search
- keeping scorer config inside the executor does not materially change repeated
  search cost relative to the direct prepared-snapshot path
- on this machine and slice, the default scorer configuration is better than
  forcing serial or forcing a two-work-item override

Why that conclusion is justified:

- the three quiet-wrapper reruns were numerically consistent even though host
  contention stayed high
- the direct and executor-backed default search means stayed within about
  `0.324%`
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
processes. The claim above is still credible because the three reruns remained
stable despite that noise, but tighter host isolation would be required for a
publication-grade absolute latency claim.

## Bottom Line

The sound next conclusion is narrow:

- the repo now has a module-level exact-search executor seam that is explicit,
  measurable, and not adding meaningful search overhead on this real slice

That is sufficient justification to keep the executor seam and build future
Python or serving concurrency work on top of it, rather than directly inside
transport code.
