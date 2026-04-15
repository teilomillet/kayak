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

- all multiples of `4` from `4` through `64`
- tail probes at `5`, `17`, `33`, and `49`

## Command

```bash
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_tiled4_query_count_sweep_browsecomp_gold.mojo
```

Artifact:

- `.cache/kayak/profile_document_proxy_stage2_tiled4_query_count_sweep_browsecomp_gold.tsv`

I did **not** wrap this benchmark with the quiet runner because it compares all
counts inside one process. Instead, I bounded each case to `16` passes over the
query set with:

- `num_warmup_iters = 0`
- `max_iters = len(task.queries) * 16`
- `min_runtime_secs = 0.0`
- `max_batch_size = 1`

That keeps the full sweep practical while still averaging over repeated passes.
It still leaves host noise, so the rollout decision below relies on clear
margins rather than tiny deltas.

## Correctness result

Every tested query-count variant matched the baseline nested dim128 stage-2
scores on the same resolved candidate windows.

That verified correctness for all counts in the sweep.

## Timing result

Measured `build_flat_query + tiled4` ratios versus the old nested dim128
baseline:

- `q=4`: `0.342211`
- `q=5`: `0.453222`
- `q=8`: `0.389353`
- `q=12`: `0.568487`
- `q=16`: `0.631724`
- `q=17`: `0.599615`
- `q=20`: `0.352722`
- `q=24`: `0.511748`
- `q=28`: `0.413419`
- `q=32`: `0.482242`
- `q=33`: `0.427526`
- `q=36`: `0.566350`
- `q=40`: `0.481204`
- `q=44`: `0.344364`
- `q=48`: `0.474734`
- `q=49`: `0.456069`
- `q=52`: `0.565841`
- `q=56`: `0.450768`
- `q=60`: `0.505157`
- `q=64`: `0.661017`

Interpreting those ratios:

- values below `1.0` are wins for the tiled kernel
- lower is better

What is clear from the sweep:

- every measured multiple of `4` from `4` through `64` won comfortably
- every sampled tail probe also won in this bounded sweep

What is **not** established by this sweep:

- that every non-multiple count between `4` and `64` also wins
- that the remainder path is uniformly good beyond the sampled tail probes

## Decision

Production exact-stage guard widened from:

- `query.vector_count == 32`

to:

- `4 <= query.vector_count <= 64`
- `query.vector_count % 4 == 0`

In other words:

- measured multiple-of-`4` counts are now enabled
- odd and other non-multiple counts still use the previous scorer

This is intentionally conservative.

I did **not** widen to every tested tail count even though `q=5`, `q=17`,
`q=33`, and `q=49` all won, because the production fast path is still a
shape-specialized `4`-lane kernel and this benchmark only samples, rather than
exhausts, the non-multiple remainder space.

## Validation after widening

Updated:

- `tests/test_collection_search_plan.mojo`

The exact-stage fast-path toggle test now compares default exact rerank against
the generic fallback across every enabled query count:

- `q = 4, 8, 12, ..., 64`

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

- the tiled exact-stage kernel generalizes cleanly across measured
  multiple-of-`4` query counts from `4` through `64`
- the broader range is correctness-checked at the exact-rerank seam
- sampled tail counts also look promising, but the full non-multiple space is
  still not exhaustively measured

Current conclusion:

- keep the widened production guard for measured multiple-of-`4` counts
- keep non-multiples on the existing scorer for now
- if more exact-stage work is needed, the next epistemic step is a focused tail
  benchmark or tail-kernel refinement rather than further broadening by
  assumption
