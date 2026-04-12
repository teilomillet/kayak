# 2026-04-13: `centroid_postings_imputed_flat` native follow-on

## Why this step

The current heavier native path in `kayak` is the WARP-inspired imputed stage:

- `centroid_postings_imputed`

That stage already changes the reduction logic relative to plain
`centroid_postings`, so the next sound question was not "is imputation good?".
That was already measured.

The sound next question was:

- can the same imputed semantics run on a tighter native layout
- with the same exact-stage faithfulness behavior
- and a better measured time profile on the existing public artifacts

This is the smallest change that isolates layout improvements from heuristic
changes in the heavier native engine family.

## What changed

New explicit stage:

- `centroid_postings_imputed_flat`

New scorer:

- `centroid_posting_imputed_flat_scores_for_segment(...)`

What it keeps identical to `centroid_postings_imputed`:

- the same centroid ranking order
- the same `bound` / `nprobe` selection logic
- the same WARP-like `t'` missing-similarity estimate
- the same token-level max reduction over selected centroid posting lists
- the same document-level sum with missing-similarity correction

What it changes:

- centroid-query similarity is computed from:
  - `flat_centroid_values`
  - flat query layout for dim-128 queries
- this follows the same native-layout direction as the earlier
  `centroid_postings_flat` follow-on

Public benchmark surfaces updated:

- `benchmarks/profile_stage1_blockmax_real_subset.mojo`
- `benchmarks/real_subset_candidate_window_sweep.mojo`
- `benchmarks/real_subset_vector_budget_sweep.mojo`
- `benchmarks/hard_recall_real_subset.mojo`

## Guardrails

Tests run:

- `pixi run test_collection_search_plan`
- `pixi run test_centroid_postings_store`

New direct equivalence guardrail:

- `test_centroid_postings_imputed_flat_stage_matches_imputed_stage_scores()`

This verifies on the toy centroid-postings fixture that the new flat imputed
scorer returns the same per-document stage-1 scores as
`centroid_postings_imputed`.

Plan-level guardrails also now exist for the new generator:

- `test_centroid_postings_imputed_flat_search_plan_exact_reranks_shortlist()`
- `test_centroid_postings_imputed_flat_search_plan_reports_oracle_miss_when_shortlist_is_too_small()`

Benchmarks were run sequentially to reduce resource contamination:

- `pixi run bench_profile_stage1_real_subset_raw`
- `pixi run bench_real_subset_candidate_windows`
- `pixi run bench_real_subset_vector_budgets`
- `pixi run bench_hard_recall_real_subset_raw`

Generated artifacts:

- `.cache/kayak/profile_stage1_blockmax_real_subset.tsv`
- `.cache/kayak/public_candidate_window_sweep.json`
- `.cache/kayak/public_vector_budget_sweep.json`
- `.cache/kayak/hard_recall_stage_aware_search.json`

## Measured outcome

### Stage-1 profile

Paired against `centroid_postings_imputed` at `candidate_k = 40`:

- `SciFact`
  - `centroid_postings_imputed`: `0.00040356414895220867 s`
  - `centroid_postings_imputed_flat`: `0.00039690506045678457 s`
  - delta: `-6.6590884954241016e-06 s`
- `FIQA`
  - `centroid_postings_imputed`: `0.0004093980877219609 s`
  - `centroid_postings_imputed_flat`: `0.0004105394957983193 s`
  - delta: `+1.1414080763584252e-06 s`
- `LIMIT-small`
  - `centroid_postings_imputed`: `0.0003384085426916881 s`
  - `centroid_postings_imputed_flat`: `0.0003333652410340512 s`
  - delta: `-5.043301657636898e-06 s`

Interpretation:

- the profile is mixed by slice
- but the flat imputed path is faster on two of the three public slices

### Candidate-window sweep

Paired against `centroid_postings_imputed` across `23` public rows:

- mean candidate-generation time delta:
  `-6.379076086956514e-06 s`
- mean candidate-recall delta: `0.0`
- recall wins / ties / losses: `0 / 23 / 0`
- time delta sign counts:
  - slower rows: `5`
  - ties: `1`
  - faster rows: `17`

Representative rows:

- `BrowseComp-Plus` gold, `candidate_k = 40`
  - time delta: `-6.150000000000003e-05 s`
  - candidate recall: `0.75` for both
- `BrowseComp-Plus` evidence, `candidate_k = 20`
  - time delta: `-3.675e-05 s`
  - candidate recall: `0.575` for both
- `FIQA`, `candidate_k = 20`
  - time delta: `-2.7499999999999964e-05 s`
  - candidate recall: `0.6333333333333334` for both

Interpretation:

- on the candidate-window public artifact, the flat imputed stage preserves
  stage-1 faithfulness exactly while improving average generation time

### Vector-budget sweep

Paired against `centroid_postings_imputed` across `100` public rows:

- mean candidate-recall delta: `0.0`
- mean `nDCG@k` delta: `0.0`
- mean recall@k delta: `0.0`
- mean success-rate delta: `0.0`
- every paired row ties on these quality metrics

Interpretation:

- on the public compressed-query / compressed-document design space,
  `centroid_postings_imputed_flat` is behaviorally identical to
  `centroid_postings_imputed`

### Hard-recall stage-aware benchmark

Paired against `centroid_postings_imputed` across the `4` hard public rows:

- mean candidate-recall delta: `0.0`
- mean `nDCG@10` delta: `0.0`
- mean recall@k delta: `0.0`
- mean success-rate delta: `0.0`
- mean search-time delta: `-8.625000000000142e-06 s`

Concrete rows:

- `BrowseComp-Plus` evidence, `candidate_k = 10`
  - time delta: `-1.3499999999999949e-05 s`
  - candidate recall: `0.4` for both
  - `nDCG@10 = 0.32181440691474994` for both
- `BrowseComp-Plus` gold, `candidate_k = 10`
  - time delta: `-1.5999999999999955e-05 s`
  - candidate recall: `0.4` for both
  - `nDCG@10 = 0.4449797343575519` for both

Interpretation:

- the flat imputed stage preserves the same measured faithfulness and judged
  quality on the hard public slice
- average total search time improves slightly

## Decision

The sound conclusion is:

- keep `centroid_postings_imputed_flat` as an explicit benchmarked generator
- treat it as the preferred current implementation shape for future
  WARP-inspired native work in `kayak`
- keep `centroid_postings_imputed` as a regression baseline rather than
  silently removing it

Why this is justified:

- unlike the earlier storage-only blockmax follow-on, this step preserves the
  current public quality metrics exactly
- unlike the earlier `centroid_postings_flat` follow-on, this step improves the
  heavier WARP-inspired family directly, which is the real next native frontier

What this does **not** justify:

- claiming a large new frontier move on its own
- claiming full WARP or GEM-style native traversal is finished

The next sound native step is therefore:

- build any heavier WARP-like traversal or pruning work on top of
  `centroid_postings_imputed_flat`
- continue judging it against:
  - exact-stage candidate recall
  - hard-recall judged quality
  - public candidate-window and vector-budget sweeps
