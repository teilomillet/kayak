# Document Proxy Stage-2 Tiled4 Flat-Query Production Rollout

## Claim

After ruling out two weaker stage-2 ideas:

- candidate-window flattening
- flat-query specialization on the existing nested scorer

the next credible exact-stage target was:

- keep the resolved candidate window as nested packed documents
- flatten the query once
- score dim128 documents with a tiled `4`-query microkernel that reuses each
  document-token load across four query vectors

This trace records the benchmark evidence, the production rollout, and the
planner-visible result.

## Why this was the next justified kernel

Earlier traces and benchmarks established:

- `document_proxy` stage-2 on `browsecomp_plus_gold` is scorer-bound
- candidate resolution and top-k assembly are negligible
- candidate-window flattening loses because the window-build tax dominates
- simple flat-query nested scoring also loses

That leaves a narrower kernel problem:

- reduce repeated document-token loads inside the exact scorer without changing
  retrieval policy or repacking the candidate window

The tiled `4`-query microkernel is the first remaining design that directly
targets that bottleneck.

## Benchmark evidence before productionization

Benchmark seam:

- `benchmarks/profile_document_proxy_stage2_flat_query_tiled4_compare_browsecomp_gold.mojo`

Wrapped pre-production benchmark logs:

- `.cache/kayak/bench_quiet/20260415T080728Z/`

The wrapper could not obtain a quiet host and force-ran both repeats under
heavy contention. Even so, the gap was large and stable enough to support a
decision.

Pre-production two-run means:

- current stage-2 scorer: `0.000686538436 s`
- `build_flat_query_dim128`: `0.000002070272 s`
- prebuilt tiled4 flat-query scorer: `0.000348500011 s`
- `build_flat_query + tiled4 scorer`: `0.000357318406 s`

Derived ratios versus the then-current scorer:

- prebuilt tiled4 scorer: `0.5076x`
  This is about `49.2%` faster.
- `build_flat_query + tiled4 scorer`: `0.5205x`
  This is about `47.9%` faster.

That was the first stage-2 experiment that remained a clear win even after the
query-build cost was included.

## Production change

Files changed:

- `kayak/scoring/maxsim.mojo`
- `kayak/planning/exact_stage.mojo`
- `tests/test_collection_search_plan.mojo`

Implementation summary:

- added a dim128 flat-query exact scorer in `maxsim.mojo`
- added a tiled `4`-query SIMD microkernel that reuses each document-token load
  across four query vectors
- wired `score_resolved_candidate_window_for_cpu` to use that kernel only for
  the measured exact-stage shape:
  - `VECTOR_SCALAR_NAME == "Float32"`
  - `vector_dim == 128`
  - `query.vector_count == 32`
  - `enable_dim128_fast_path == True`

The explicit `query.vector_count == 32` guard is deliberate. The benchmark
evidence in this loop was for the `q=32` shape, so the production rollout stays
 inside that measured regime instead of assuming the win generalizes.

## Correctness validation

Commands:

```bash
pixi run mojo -I . tests/test_maxsim.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_flat_query_tiled4_compare_browsecomp_gold.mojo
pixi run mojo -I . benchmarks/profile_document_proxy_exact_stage_breakdown_browsecomp_gold.mojo
pixi run mojo -I . benchmarks/profile_document_proxy_search_breakdown_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --repeats 2 --timeout-seconds 20 --force -- pixi run mojo -I . benchmarks/profile_document_proxy_stage2_flat_query_tiled4_compare_browsecomp_gold.mojo
```

Test results:

- `tests/test_maxsim.mojo`: `7/7` passed
- `tests/test_collection_search_plan.mojo`: `39/39` passed

New regression guard:

- `test_exact_rerank_candidates_for_plan_dim128_fast_path_toggle_keeps_scores_identical_for_resolved_hits`

That test matters because it compares the new exact-stage fast path against the
generic fallback at the resolved-hit rerank seam, which is exactly where this
optimization was inserted.

## Production benchmark result

Wrapped post-production benchmark logs:

- `.cache/kayak/bench_quiet/20260415T082406Z/`

Again, both repeats force-ran under host contention. The direction remained
stable.

Post-production two-run means from the same benchmark seam:

- current stage-2 scorer: `0.000352463309 s`
- `build_flat_query_dim128`: `0.000002078936 s`
- prebuilt tiled4 flat-query scorer: `0.000346734915 s`
- `build_flat_query + tiled4 scorer`: `0.000352310739 s`

Key comparison:

- production current before rollout: `0.000686538436 s`
- production current after rollout: `0.000352463309 s`

Derived ratio:

- post / pre: `0.513392`
- speedup: about `1.95x`

This is the important production result: the benchmark-local win survived the
integration. The live `score_resolved_candidate_window_for_cpu` path now lands
in the same band as the benchmark-local tiled4 helper.

## Planner-visible stage result

Current exact-stage breakdown:

- source: `.cache/kayak/profile_document_proxy_exact_stage_breakdown_browsecomp_gold.tsv`
- `resolve_candidate_window`: `8.313642667044171e-07 s`
- `score_resolved_candidate_window_for_cpu`: `0.00033684030701754386 s`
- `assemble_topk_hits`: `5.202856258696874e-07 s`

Current document-proxy search breakdown:

- source: `.cache/kayak/profile_document_proxy_search_breakdown_browsecomp_gold.tsv`
- `stage1_document_proxy`: `1.734398452546988e-05 s`
- `stage2_from_candidates`: `0.0003322875225281864 s`
- `search_document_proxy`: `0.00035877260286265535 s`

Previous measured baseline from:

- `docs/traces/2026-04-15_document_proxy_stage2_gold_investigation.md`

Previous means on the same slice:

- `score_resolved_candidate_window_for_cpu`: `0.000689044106167057 s`
- `stage2_from_candidates`: `0.0006926189695550351 s`
- `search_document_proxy`: `0.0007375159010600706 s`

Derived stage-level deltas:

- exact-stage scorer ratio: `0.488857`
  This is about `2.05x` faster.
- `stage2_from_candidates` ratio: `0.479755`
  This is about `2.08x` faster.
- `search_document_proxy` ratio: `0.486461`
  This is about `2.06x` faster.

Stage1 stayed essentially unchanged, which is what we would expect from a
pure stage-2 scorer optimization.

## Bottom line

What is now verified:

- the earlier rejected stage-2 ideas were correctly rejected
- the tiled `4`-query flat-query microkernel is the first measured stage-2
  scorer change that wins both in isolation and after production integration
- on `document_proxy` BrowseComp gold, the production exact-stage scorer is now
  about `1.95x` faster on the wrapped benchmark seam
- the planner-visible `stage2_from_candidates` and whole
  `search_document_proxy` path are both about `2.06x` faster on the current
  benchmark seams

Current conclusion:

- keep this production change
- keep the explicit shape guard at `q=32` until more query-count shapes are
  benchmarked
- if more stage-2 work is needed, the next epistemic step is to test whether
  this tiled kernel generalizes to additional query-vector counts rather than
  broadening the guard on assumption
