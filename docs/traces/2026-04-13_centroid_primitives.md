# 2026-04-13: centroid selection and accumulation primitives

## Why this step

The centroid-family stage-1 generators in `kayak` had converged on the same
two underlying operations:

- score a query token against representative centroids and keep an ordered
  shortlist
- accumulate posting-list contributions into per-document token-max scores

Before this step, those operations were reimplemented in several stage files:

- `centroid_postings`
- `centroid_postings_flat`
- `centroid_postings_imputed`
- `centroid_postings_imputed_flat`

That duplication made the engine harder to browse, harder to profile, and
harder to evolve without accidental semantic drift.

The sound question was therefore not "can we invent another heuristic?" but:

- can we extract the shared late-interaction-native primitives directly
- keep the public retrieval behavior unchanged
- and use those primitives to identify the next real CPU hot path

## What changed

New primitive module:

- [kayak/planning/centroid_primitives.mojo](../../kayak/planning/centroid_primitives.mojo)

New exported primitive surface:

- `ScoredCentroidSelection`
- `accumulate_selected_centroid_scores(...)`

What this primitive layer represents:

- `ScoredCentroidSelection`
  - the ordered centroid shortlist for one query token
  - the centroid scores needed for posting accumulation
  - the optional baseline correction used by imputed stages
- `accumulate_selected_centroid_scores(...)`
  - the shared posting traversal and per-document token-max accumulation step

Refactored stage entry points now build on that shared layer:

- [kayak/planning/centroid_postings_stage.mojo](../../kayak/planning/centroid_postings_stage.mojo)
- [kayak/planning/centroid_postings_flat_stage.mojo](../../kayak/planning/centroid_postings_flat_stage.mojo)
- [kayak/planning/centroid_postings_imputed_stage.mojo](../../kayak/planning/centroid_postings_imputed_stage.mojo)
- [kayak/planning/centroid_postings_imputed_flat_stage.mojo](../../kayak/planning/centroid_postings_imputed_flat_stage.mojo)

New primitive-level benchmark:

- [benchmarks/profile_centroid_primitives_real_subset.mojo](../../benchmarks/profile_centroid_primitives_real_subset.mojo)

The benchmark writes:

- `.cache/kayak/profile_centroid_primitives_real_subset.tsv`

## Guardrails

Tests rerun after the refactor:

- `pixi run test_centroid_postings_store`
- `pixi run test_collection_search_plan`

Important semantic guardrail already in-tree:

- `test_centroid_postings_imputed_flat_stage_matches_imputed_stage_scores()`

This keeps the refactor honest on a direct toy fixture: the flat imputed stage
must still produce the same stage-1 scores as the non-flat imputed stage.

Public artifact reruns after the refactor:

- `pixi run bench_real_subset_candidate_windows`
- `pixi run bench_real_subset_vector_budgets`
- `pixi run bench_hard_recall_real_subset_raw`

Generated artifacts:

- `.cache/kayak/public_candidate_window_sweep.json`
- `.cache/kayak/public_vector_budget_sweep.json`
- `.cache/kayak/hard_recall_stage_aware_search.json`

## Verified semantic outcome

The important post-refactor result is semantic stability.

### Candidate-window artifact

Paired `centroid_postings_imputed_flat` against `centroid_postings_imputed`
across `23` public rows:

- mean candidate-recall delta: `0.0`
- recall wins / ties / losses: `0 / 23 / 0`

The current raw candidate-generation time mean still favors the flat path by a
small margin:

- mean candidate-generation delta: `-3.317481884057978e-06 s`

But this micro-delta is not the main decision signal for this step.

### Vector-budget artifact

Paired across `100` public rows:

- mean candidate-recall delta: `0.0`
- mean `nDCG@k` delta: `0.0`
- mean recall@k delta: `0.0`
- mean success-rate delta: `0.0`

Every paired row ties on these quality metrics.

### Hard-recall artifact

Paired across the `4` hard public rows:

- mean candidate-recall delta: `0.0`
- mean `nDCG@10` delta: `0.0`
- mean recall@k delta: `0.0`
- mean success-rate delta: `0.0`

Interpretation:

- the primitive extraction did not change the measured retrieval behavior of
  the heavier imputed family on the public hard-recall slice

## Primitive benchmark result

The primitive benchmark is the real reason this refactor matters.

Measured on `SciFact`, `FIQA`, and `LIMIT-small`:

- imputed selection is the dominant primitive cost
- accumulation is materially cheaper than selection
- the flat imputed selector is consistently faster than the nested imputed
  selector on all three slices

Representative values:

- `SciFact`
  - `imputed_selection_nested`: `0.0003324574876104288 s`
  - `imputed_selection_flat`: `0.0003253436793422405 s`
  - `imputed_accumulation`: `6.817946584488957e-05 s`
- `FIQA`
  - `imputed_selection_nested`: `0.00033497504201680675 s`
  - `imputed_selection_flat`: `0.00033028929612661395 s`
  - `imputed_accumulation`: `7.537080166707527e-05 s`
- `LIMIT-small`
  - `imputed_selection_nested`: `0.00028312907103825136 s`
  - `imputed_selection_flat`: `0.0002750530744336569 s`
  - `imputed_accumulation`: `6.432609931127487e-05 s`

Ratios from the measured artifact:

- imputed selection is about `4.4x` to `4.9x` more expensive than imputed
  accumulation
- imputed selection is about `4.7x` to `5.0x` more expensive than exact
  centroid selection

Inference:

- the next CPU optimization target should be the imputed centroid-selection
  kernel
- it is not sound to spend another optimization cycle on accumulation first

## Timing caveat

I attempted to rerun the primitive benchmark through the quiet-host wrapper:

- `bash scripts/run_bench_quiet.sh -- pixi run bench_profile_centroid_primitives_real_subset_raw`

The wrapper behaved correctly, but the host never reached the default quiet
threshold. Observed competing CPU stayed around `300-650%`, dominated by local
desktop/editor/system activity such as:

- `Zed.app`
- macOS storage-management services
- `WindowServer`
- `modular-crashpad-handler`

Decision:

- treat the semantic-equality results above as the hard guardrail
- treat the primitive benchmark as the hotspot-identification evidence
- do **not** overclaim from tiny wall-clock deltas on a busy host

## Conclusion

This step is complete and justified.

What is now true:

- centroid selection and posting accumulation are first-class engine
  primitives in `kayak`
- the centroid-family stage implementations are materially easier to browse and
  profile
- public retrieval behavior stayed stable on the measured artifacts
- the next optimization target is now clearer: imputed selection, not
  accumulation

What this does **not** mean:

- centroids are the fundamental retrieval primitive
- WARP or GEM-style native traversal is finished

The correct interpretation is narrower:

- the centroid/imputation family is now expressed in reusable engine
  primitives
- those primitives give us a sound base for the next heavier native step
