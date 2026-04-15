# Document Proxy Stage-2 Tiled4 Query-Count Sweep

## Claim

After shipping the tiled `4`-query flat-query microkernel for the measured
`q=32` exact-stage shape, the next question was:

- does the same kernel generalize to other query-vector counts strongly enough
  to widen the production guard

This trace records the benchmark used to answer that question and the rollout
decision that followed.

## Why this check was needed

The previous production trace verified a clear planner-visible win for:

- `vector_dim = 128`
- `query.vector_count = 32`

But the exact-stage guard was intentionally narrow because that was the only
shape actually benchmarked at rollout time.

Before broadening the guard, I needed evidence for two things:

- whether the tiled kernel still wins when `query.vector_count != 32`
- whether the tail path for non-multiples of `4` is also safe to enable

## Constraint

All real cached retrieval slices in this repo currently use:

- `nominal_query_vector_count = 32`

Verified from the stored task manifests under:

- `.cache/kayak/scifact_real_subset/judged_task/manifest.tsv`
- `.cache/kayak/fiqa_real_subset/judged_task/manifest.tsv`
- `.cache/kayak/limit_small_real_subset/judged_task/manifest.tsv`
- `.cache/kayak/browsecomp_plus_real_subset/judged_task/manifest.tsv`
- `.cache/kayak/browsecomp_plus_gold_real_subset/judged_task/manifest.tsv`

So query-count generalization beyond `32` could not be tested as a planner-level
real-workload benchmark inside this repo. It had to be tested as a stage-2
microkernel study on real candidate windows with synthetic query-count variants.

That limitation matters and is reflected in the rollout decision below.

## Benchmark seam

Added:

- `benchmarks/profile_document_proxy_stage2_tiled4_query_count_sweep_browsecomp_gold.mojo`

The benchmark:

- reuses the real BrowseComp-Plus gold `document_proxy` resolved candidate
  windows
- compares the old nested dim128 exact scorer against the tiled flat-query
  scorer on the same windows
- validates exact score equality at every tested query count
- measures both:
  - prebuilt tiled scorer
  - `build_flat_query + tiled scorer`

Synthetic query-count variants are constructed as:

- `q <= 32`: prefix of the real query vectors
- `q > 32`: cyclic extension of the real query vectors

That means:

- document windows are real
- query values are still real at the source-vector level
- counts above `32` are synthetic extensions, not planner-produced workloads

Measured fixed candidate-window shape:

- mean candidate documents: `40.0`
- mean candidate vectors: `6974.75`

Measured query counts:

- every `q` from `4` through `64`

## Command

```bash
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_tiled4_query_count_sweep_browsecomp_gold.mojo
```

Artifact:

- `.cache/kayak/profile_document_proxy_stage2_tiled4_query_count_sweep_browsecomp_gold.tsv`

I did **not** wrap this benchmark with the quiet runner because it compares all
counts inside one process. Instead, I bounded each case with:

- `num_warmup_iters = 0`
- `max_iters = len(task.queries) * 16 * max(1, 64 // q)`
- `min_runtime_secs = 0.0`
- `max_batch_size = 1`

The `max(1, 64 // q)` scale factor matters. An earlier fixed-iteration version
under-ran the smallest query counts and produced unstable low-`q` results on a
busy host. The normalized schedule keeps total work much closer across counts,
which is the version used for the rollout decision below.

## Correctness result

Every tested query-count variant matched the baseline nested dim128 stage-2
scores on the same resolved candidate windows.

That verified correctness for all counts in the sweep.

## Timing result

Measured `build_flat_query + tiled4` ratios versus the old nested dim128
baseline still stayed below `1.0` for every count in `4..64`.

Representative points:

- `q=5`: `0.337015`
- `q=7`: `0.611198`
- `q=19`: `0.574508`
- `q=26`: `0.711688`
- `q=33`: `0.426296`
- `q=49`: `0.521711`
- `q=63`: `0.618680`

Worst measured tail ratio:

- `q=26`: `0.711688`

Best measured tail ratio:

- `q=5`: `0.337015`

Interpreting those ratios:

- values below `1.0` are wins for the tiled kernel
- lower is better

What is clear from the sweep:

- every measured count in `4..64` won in the build-inclusive path
- the previous tail ambiguity was a benchmark-design artifact from
  under-running the smallest query counts, not a stable remainder-path loss

## Decision

Production exact-stage guard widened from:

- `query.vector_count == 32`

to:

- `4 <= query.vector_count <= 64`

In other words:

- every measured count in `4..64` is now enabled
- counts outside that range still use the previous scorer

## Validation after widening

Updated:

- `tests/test_collection_search_plan.mojo`

The exact-stage fast-path toggle test now compares default exact rerank against
the generic fallback across every enabled query count:

- `q = 4, 5, 6, ..., 64`

Command:

```bash
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_maxsim.mojo
```

Results:

- `tests/test_collection_search_plan.mojo`: `39/39` passed
- `tests/test_maxsim.mojo`: `7/7` passed

## Bottom line

What is now verified:

- the tiled exact-stage kernel generalizes cleanly across all measured query
  counts from `4` through `64`
- the broader range is correctness-checked at the exact-rerank seam

Current conclusion:

- keep the widened production guard for all measured counts in `4..64`
- if more exact-stage work is needed after this, the next epistemic step is a
  new kernel idea with a benchmark that beats the current tiled4 path, not more
  rollout work
