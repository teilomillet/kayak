# Exact Rerank Parallel Dispatch on BrowseComp-Plus Gold

## Claim

After the earlier traces established that:

- stage-2 exact rerank dominated whole-search time on `browsecomp_plus_gold`
- per-query candidate-window materialization dominated inside exact rerank

the next claim to test was:

- can exact stage-2 become materially faster by keeping the no-repack CPU path,
  parallelizing direct candidate scoring, and making sure the planner actually
  dispatches into that specialized path

This trace records the code inspection, the correction, and the measured result.

## Code inspection that motivated the change

I inspected the exact-stage and planner dispatch surfaces:

- [exact_stage.mojo](/Users/teilomillet/Code/kayak/kayak/planning/exact_stage.mojo)
- [execution_stage2.mojo](/Users/teilomillet/Code/kayak/kayak/planning/execution_stage2.mojo)
- [execution.mojo](/Users/teilomillet/Code/kayak/kayak/planning/execution.mojo)
- [explain.mojo](/Users/teilomillet/Code/kayak/kayak/planning/explain.mojo)
- [maxsim.mojo](/Users/teilomillet/Code/kayak/kayak/scoring/maxsim.mojo)

What was true before this slice:

- the repo already had a no-repack `ExactCpuBackend` rerank path in
  `exact_stage.mojo`
- that path still scored resolved candidate documents serially
- planner and search entry points were generic over
  `Backend: ExactScoringBackend`

The first direct exact-stage rerun showed a large microbenchmark win, but the
first search-breakdown rerun did not inherit it. That mismatch mattered.

The most plausible code-level explanation was:

- direct calls to `exact_rerank_candidates_for_plan(ExactCpuBackend, ...)`
  reached the optimized CPU overload
- generic planner dispatch could still resolve through the trait-constrained
  path instead of the `ExactCpuBackend`-specific path

That interpretation matched the measurements and was then fixed explicitly.

## Implementation

The final change set did three things:

- reused the existing no-repack candidate resolution path, but changed exact
  CPU rerank to score the resolved candidate window in parallel and maintain
  only the final top-k once
- lifted the maxsim parallel work-item policy into a shape-based helper so the
  exact-stage scorer and packed-index scorer use the same basic parallelism rule
- added `ExactCpuBackend` overloads for planner dispatch in:
  - [execution_stage2.mojo](/Users/teilomillet/Code/kayak/kayak/planning/execution_stage2.mojo)
  - [execution.mojo](/Users/teilomillet/Code/kayak/kayak/planning/execution.mojo)
  - [explain.mojo](/Users/teilomillet/Code/kayak/kayak/planning/explain.mojo)

That last part is important. Without it, the optimized stage-2 kernel existed
locally but was not reliably reachable from the public planned-search path.

Validation added:

- [test_collection_search_plan.mojo](/Users/teilomillet/Code/kayak/tests/test_collection_search_plan.mojo)
  now forces a resolved-hit rerank down both serial and parallel exact CPU
  paths and checks that the results match

## Commands run

```bash
pixi run mojo -I . tests/test_maxsim.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . benchmarks/profile_exact_stage_materialization_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 1 -- pixi run mojo -I . benchmarks/profile_exact_stage_materialization_browsecomp_gold.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 1 -- pixi run mojo -I . benchmarks/profile_centroid_workspace_search_breakdown.mojo
```

Correctness checks passed:

- `tests/test_maxsim.mojo`: `7/7`
- `tests/test_collection_search_plan.mojo`: `38/38`
- `tests/test_service_runtime.mojo`: `26/26`

Quiet-wrapper artifacts:

- `.cache/kayak/bench_quiet/20260414T215231Z`
- `.cache/kayak/bench_quiet/20260414T221505Z`

Both wrapped runs timed out waiting for a quiet host and then force-ran under
contention. That is still comparable to the earlier wrapped baselines, which
used the same wrapper under similar host load.

## Wrapped exact-stage result

Previous wrapped exact-stage baseline, before this parallel direct-score slice:

- source: `.cache/kayak/bench_quiet/20260414T212725Z/run_1.txt`
- `centroid_postings_imputed`
  - `exact_rerank_candidates_for_plan`: `2.1899653739612186e-03 s`
- `centroid_postings_imputed_flat`
  - `exact_rerank_candidates_for_plan`: `2.1977450800915333e-03 s`

Current wrapped exact-stage result:

- source: `.cache/kayak/profile_exact_stage_materialization_browsecomp_gold.tsv`
- `centroid_postings_imputed`
  - `materialize_candidate_index`: `1.3188710193398285e-03 s`
  - `score_materialized_index`: `9.423777746459024e-04 s`
  - `exact_rerank_candidates_for_plan`: `6.472804232804233e-04 s`
- `centroid_postings_imputed_flat`
  - `materialize_candidate_index`: `1.336339188021883e-03 s`
  - `score_materialized_index`: `7.632755453501722e-04 s`
  - `exact_rerank_candidates_for_plan`: `1.0077309273182958e-03 s`

Derived deltas:

- `centroid_postings_imputed`
  - exact rerank reduction: `-1.542685 ms`
  - exact rerank ratio: `0.295566`
  - speedup: `3.38x`
- `centroid_postings_imputed_flat`
  - exact rerank reduction: `-1.190014 ms`
  - exact rerank ratio: `0.458535`
  - speedup: `2.18x`

Two details matter here:

- the optimized no-repack rerank is now faster than materializing the candidate
  window at all
- for the plain imputed plan, the direct rerank is now also below the wrapped
  `score_materialized_index` timing

That is a qualitatively different regime from the earlier trace, where rerank
was dominated by materialization plus remaining overhead.

## Wrapped search-breakdown result

Previous wrapped search-breakdown baseline:

- source:
  [2026-04-14_centroid_workspace_search_breakdown_browsecomp_gold.md](/Users/teilomillet/Code/kayak/docs/traces/2026-04-14_centroid_workspace_search_breakdown_browsecomp_gold.md)
- `centroid_postings_imputed`
  - stage2 from candidates: `2.2539228874106925e-03 s`
  - search fresh: `2.8110515071700323e-03 s`
  - search reused: `3.0684335622573906e-03 s`
- `centroid_postings_imputed_flat`
  - stage2 from candidates: `2.2942761834682363e-03 s`
  - search fresh: `2.6707811027985593e-03 s`
  - search reused: `2.7809116465863453e-03 s`

Current wrapped search-breakdown result:

- source: `.cache/kayak/profile_centroid_workspace_search_breakdown.tsv`
- `centroid_postings_imputed`
  - stage1 fresh: `4.4192139232127097e-04 s`
  - stage1 reused: `4.44796556851979e-04 s`
  - stage2 from candidates: `6.976382375928097e-04 s`
  - search fresh: `1.0415690058479532e-03 s`
  - search reused: `1.0147310588235294e-03 s`
- `centroid_postings_imputed_flat`
  - stage1 fresh: `4.275133426966292e-04 s`
  - stage1 reused: `4.3580412705272254e-04 s`
  - stage2 from candidates: `6.040420347677401e-04 s`
  - search fresh: `1.000669799426934e-03 s`
  - search reused: `9.963423121387284e-04 s`

Derived deltas:

- `centroid_postings_imputed`
  - stage2 reduction: `-1.556285 ms`
  - stage2 ratio: `0.309522`
  - stage2 speedup: `3.23x`
  - fresh-search reduction: `-1.769483 ms`
  - fresh-search ratio: `0.370524`
  - fresh-search speedup: `2.70x`
- `centroid_postings_imputed_flat`
  - stage2 reduction: `-1.690234 ms`
  - stage2 ratio: `0.263278`
  - stage2 speedup: `3.80x`
  - fresh-search reduction: `-1.670111 ms`
  - fresh-search ratio: `0.374677`
  - fresh-search speedup: `2.67x`

## Interpretation

What is verified:

- the exact CPU no-repack rerank kernel became much faster in isolation
- the planner-visible search path now inherits that win after the dispatch fix
- on this slice, the improvement is no longer a microbenchmark-only result
- whole-search latency drops by roughly `63%` on both imputed centroid plans in
  the wrapped search benchmark

What changed in the stage split:

- before this slice, stage2 was roughly `5x` the cost of stage1 on these plans
- after this slice, stage2 is still the largest single stage, but only about
  `1.4x` to `1.6x` stage1 in the wrapped run

That means the earlier diagnosis was correct:

- stage2 exact rerank really was the dominant whole-search bottleneck
- fixing it moved the end-to-end latency materially

It also means the next bottleneck has changed:

- stage1 candidate generation is no longer dwarfed by stage2 on this workload
- if more whole-search wins are needed from here, they should probably target
  stage1 centroid work or another planner-visible subsystem, not more micro
  tuning inside the already-fixed rerank path

## Bottom line

The strongest supported reading is now:

- exact rerank materialization and dispatch were the right next targets
- parallel direct scoring plus explicit `ExactCpuBackend` planner dispatch
  turned a local kernel win into a real whole-search win
- on `browsecomp_plus_gold`, wrapped fresh search improved by about `2.7x`
  for both imputed centroid plans

That is a measured planner-visible speedup, not just a constant-factor
microbenchmark improvement hidden under the API seam.
