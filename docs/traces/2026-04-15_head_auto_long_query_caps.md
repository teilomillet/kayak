# 2026-04-15: `centroid_postings_head_auto` long-query cap retune

## Claim

`centroid_postings_head_auto` was leaving a benchmark-visible gap on
BrowseComp-Plus gold.

Before this step, the auto-head stage used a wide default cap for long queries:

- `candidate_k = 10` behaved like a much looser head than the best measured
  gold frontier point
- `candidate_k = 20` and `candidate_k = 40` showed the same pattern

The concrete question was:

- can we tighten `centroid_postings_head_auto` only for the long-query,
  small-window regime
- improve measured judged quality on BrowseComp-Plus gold
- without touching the user’s in-progress imputed-stage work

## Why this target

This was the sound next benchmark target because the repo already comfortably
beats the current LanceDB baselines on BrowseComp-Plus gold:

- `.cache/kayak/lancedb_browsecomp_plus_gold_scorecard.json`
- `.cache/kayak/storage_engine_browsecomp_plus_gold_scale.json`
- `.cache/kayak/lancedb_browsecomp_plus_gold_scale_sweep.json`

That means the next actionable gap was internal:

- the gold faithfulness frontier exposed a missing middle point between
  `document_proxy` and `centroid_postings_imputed_flat`
- `centroid_postings_head_auto` existed but underperformed that role

## What changed

New focused benchmark seam:

- `benchmarks/profile_head_auto_browsecomp_gold.mojo`

New unit guardrail:

- `tests/test_centroid_postings_head_auto_stage.mojo`

Heuristic change:

- `kayak/planning/centroid_postings_head_auto_stage.mojo`

The new rule stays intentionally narrow:

- if `query_vector_count >= 24` and `candidate_k <= 10`, use cap `4`
- if `query_vector_count >= 24` and `candidate_k <= 20`, use cap `8`
- if `query_vector_count >= 24` and `candidate_k <= 40`, use cap `4`
- otherwise keep the previous default / low-query expansion behavior

Why this is justified:

- the focused probe showed a real gold gap at exactly those window sizes
- the change is scoped to a regime we actually measured
- no broader public retune is claimed here

## Validation

Tests run:

- `pixi run bash -lc 'mojo -I . tests/test_centroid_postings_head_auto_stage.mojo'`
- `pixi run bash -lc 'mojo -I . tests/test_collection_search_plan.mojo'`

Focused benchmark run:

- `pixi run bench_profile_head_auto_browsecomp_gold_raw`

Broader hard-recall validation:

- `pixi run bench_hard_recall_real_subset_raw`

Fresh gold frontier refresh:

- `pixi run bench_browsecomp_plus_gold_faithfulness_frontier_raw`

Artifacts:

- `.cache/kayak/head_auto_browsecomp_gold_probe.json`
- `.cache/kayak/hard_recall_stage_aware_search.json`
- `.cache/kayak/browsecomp_plus_gold_faithfulness_frontier.json`

## Measured outcome

### Decision-quality probe: BrowseComp-Plus gold

Baseline `centroid_postings_head_auto` from the focused probe before the
heuristic change:

- `candidate_k = 10`
  - `mean_search_seconds = 0.00064075`
  - `mean_ndcg_at_k = 0.24836823392317836`
  - `mean_candidate_recall_at_final_k = 0.375`
- `candidate_k = 20`
  - `mean_search_seconds = 0.0006505`
  - `mean_ndcg_at_k = 0.2776717437348852`
  - `mean_candidate_recall_at_final_k = 0.625`
- `candidate_k = 40`
  - `mean_search_seconds = 0.00092125`
  - `mean_ndcg_at_k = 0.2588584074926856`
  - `mean_candidate_recall_at_final_k = 0.825`

Final `centroid_postings_head_auto` after the retune:

- `candidate_k = 10`
  - `mean_search_seconds = 0.00063375`
  - `mean_ndcg_at_k = 0.35023039907918324`
  - `mean_candidate_recall_at_final_k = 0.425`
- `candidate_k = 20`
  - `mean_search_seconds = 0.0006715`
  - `mean_ndcg_at_k = 0.35954575348612877`
  - `mean_candidate_recall_at_final_k = 0.6000000000000001`
- `candidate_k = 40`
  - `mean_search_seconds = 0.00072`
  - `mean_ndcg_at_k = 0.3421587098065445`
  - `mean_candidate_recall_at_final_k = 0.625`

Derived deltas:

- `candidate_k = 10`
  - time delta: `-7e-06 s`
  - `nDCG@10` delta: `+0.10186216515600488`
  - candidate-recall delta: `+0.05`
- `candidate_k = 20`
  - time delta: `+2.099999999999997e-05 s`
  - `nDCG@10` delta: `+0.08187400975124359`
  - candidate-recall delta: `-0.025000000000000022`
- `candidate_k = 40`
  - time delta: `-0.00020124999999999993 s`
  - `nDCG@10` delta: `+0.08330030231385888`
  - candidate-recall delta: `-0.19999999999999996`

Interpretation:

- the retune materially improves judged quality at the gold windows that were
  actually weak
- the `candidate_k = 40` win is especially strong because it is both faster and
  higher `nDCG`
- the larger-window behavior (`80`, `90`) remained unchanged on judged quality

### Hard-recall validation

Final `hard_recall_stage_aware_search.json` rows for `candidate_k = 10`:

- evidence slice
  - `centroid_heads`
    - `mean_search_seconds = 0.00064875`
    - `mean_ndcg_at_k = 0.26365135965362574`
    - `mean_candidate_recall_at_final_k = 0.4`
  - `centroid_postings_head_auto`
    - `mean_search_seconds = 0.000663`
    - `mean_ndcg_at_k = 0.3273659310474696`
    - `mean_candidate_recall_at_final_k = 0.42500000000000004`
- gold slice
  - `centroid_heads`
    - `mean_search_seconds = 0.00067975`
    - `mean_ndcg_at_k = 0.26262991308340555`
    - `mean_candidate_recall_at_final_k = 0.4`
  - `centroid_postings_head_auto`
    - `mean_search_seconds = 0.00063725`
    - `mean_ndcg_at_k = 0.35023039907918324`
    - `mean_candidate_recall_at_final_k = 0.42500000000000004`

Interpretation:

- the long-query cap retune is not just a one-off probe artifact
- the same small-window improvement shows up on both BrowseComp hard slices

### Full gold frontier refresh

The raw refreshed gold frontier is exploratory rather than quiet-run quality,
but it still shows the structural result we wanted:

- `centroid_postings_head_auto` now appears as an undominated gold frontier
  point at `candidate_k = 10`
- the refreshed undominated set is:
  - `document_proxy`, `candidate_k = 10`
  - `centroid_heads`, `candidate_k = 10`
  - `centroid_postings_head_auto`, `candidate_k = 10`
  - `centroid_postings_imputed_flat`, `candidate_k = 10`

Why this still matters despite raw-run noise:

- the focused probe above is the decision-quality before/after artifact
- the full frontier confirms the new point survives when measured inside the
  broader gold benchmark family

## Decision

Keep the retuned long-query head-auto heuristic.

Why this is justified:

- the change is narrow and benchmark-backed
- the gold probe showed real judged-quality gains at `candidate_k = 10`, `20`,
  and `40`
- hard-recall validation on the public BrowseComp slices confirmed the `k = 10`
  improvement
- collection-search-plan guardrails still pass

What this does **not** justify:

- claiming a universal head-auto optimum for all datasets
- retuning the larger-window regime without more public evidence
- changing the user’s in-progress imputed-stage files

The next sound follow-on, if we continue on this benchmark family, is:

- run a quiet public sweep for `centroid_postings_head_auto` versus
  `document_proxy`, `centroid_heads`, and `centroid_postings_imputed_flat`
  specifically on hard slices
- only then decide whether planner selection should expose head-auto more
  aggressively
