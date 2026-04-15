# Centroid Exact Stage Top-K Fusion Negative Result

## Claim

After the earlier exact-stage work removed candidate-window repacking and the
new centroid breakdown benchmark showed that:

- `resolve_candidate_window(...)` is negligible on the current BrowseComp gold
  centroid path
- `assemble_topk_hits(...)` is also negligible
- the remaining exact-stage time is almost entirely the scorer path plus
  whatever score materialization sits around it

the next claim to test was:

- can exact rerank become faster by scoring directly into bounded top-k state
  instead of materializing a full score list and then scanning it again

This trace records the benchmark seam, the attempted production change, the
measured result, and the reversion.

## Why this was the right hypothesis to test

I first added a current-code centroid exact-stage breakdown benchmark:

- [benchmarks/profile_centroid_exact_stage_breakdown_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_centroid_exact_stage_breakdown_browsecomp_gold.mojo)

That benchmark measures, on the real BrowseComp-Plus gold centroid shortlist:

- `resolve_candidate_window`
- `score_resolved_candidate_window_for_cpu`
- `assemble_topk_hits`
- `exact_rerank_candidates_for_plan`

The first run on the current code established the shape of the remaining cost:

- `centroid_postings_imputed`
  - `resolve_candidate_window`: about `0.85` to `0.89 µs`
  - `assemble_topk_hits`: about `0.58` to `0.62 µs`
  - `score_resolved_candidate_window_for_cpu`: about `303` to `315 µs`

So the bookkeeping-oriented explanation was falsified for this workload.

That left one narrower idea that was still worth testing:

- keep exact scoring semantics unchanged
- remove the full score-vector materialization from the CPU exact rerank path
- maintain only bounded top-k state while scoring

## Attempted production change

I implemented a temporary production experiment in
[kayak/planning/exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo):

- score resolved candidate windows directly into per-partition bounded top-k
  buffers
- merge those bounded results instead of building a full `List[ScoreScalar]`
- use the same exact scorer kernels as before, including the measured dim128
  tiled path

The intended reason was explicit:

- if score-list materialization was a meaningful remaining tax, this change
  should beat the old `resolve -> score list -> assemble top-k` path on the
  same shortlist benchmark

## Benchmark seam added for the A/B

The same benchmark was then extended with an explicit reference path:

- `exact_rerank_full_score_reference`

That benchmark-local reference does:

1. `resolve_candidate_window(...)`
2. `score_resolved_candidate_window_for_cpu(...)`
3. `assemble_topk_hits(...)`

This matters because it compares the old and new exact-rerank structures
inside the same binary and the same host run instead of relying on two
separate runs stitched together later.

## Commands run

```bash
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_maxsim.mojo
pixi run mojo -I . benchmarks/profile_centroid_exact_stage_breakdown_browsecomp_gold.mojo
```

Correctness checks passed both before and after the revert:

- `tests/test_maxsim.mojo`: `7/7`
- `tests/test_collection_search_plan.mojo`: `39/39`

## In-process A/B result

On the run with the temporary fused top-k production path still enabled, the
benchmark reported:

- `centroid_postings_imputed`
  - `exact_rerank_full_score_reference`: `0.0003107422178185121 s`
  - `exact_rerank_candidates_for_plan`: `0.00036098664389560344 s`
- `centroid_postings_imputed_flat`
  - `exact_rerank_full_score_reference`: `0.0003494876921709766 s`
  - `exact_rerank_candidates_for_plan`: `0.0005164704357943728 s`

Derived ratios:

- `centroid_postings_imputed`
  - new / reference: `1.1617x`
  - slowdown: about `16.2%`
- `centroid_postings_imputed_flat`
  - new / reference: `1.4778x`
  - slowdown: about `47.8%`

That is not ambiguous. The attempted optimization lost badly on the exact seam
it was supposed to improve.

## Decision

Revert the production fused top-k change.

Reason:

- the benchmark-local A/B directly debunked the hypothesis
- keeping the change would knowingly ship a regression on the measured
  centroid stage-2 path
- the benchmark itself is valuable and should remain in the repo

After the revert, the benchmark still compiles and the production tests still
pass. The production exact-stage path stays on the earlier direct-score design.

## What is now verified

- on the current BrowseComp gold centroid shortlist, candidate resolution is
  not the remaining bottleneck
- final top-k assembly is also not the remaining bottleneck
- a straightforward “score directly into bounded top-k state” rewrite is a
  measured negative result here
- the new benchmark seam is worth keeping because it can distinguish real wins
  from plausible-but-wrong hot-path ideas

## Current conclusion

The remaining centroid exact-stage frontier is not another bookkeeping rewrite.
If more stage-2 speed is needed from here, the next credible move has to be a
genuinely different scorer or layout strategy, not a new way to maintain the
same top-k state.
