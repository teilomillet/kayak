# BrowseComp-Plus Gold Frontier Includes `centroid_postings_imputed_flat`

## Claim

The gold faithfulness frontier should include
`centroid_postings_imputed_flat`, not only `centroid_postings_imputed`.

Reason:

- the planner registry already promotes `centroid_postings_imputed_flat`
- `centroid_postings_imputed` is currently kept as a regression baseline
- the public frontier should therefore expose the current preferred native
  WARP-shaped path when reporting Kayak's approximate surface

## Code inspected

- [planner_registry.mojo](../../kayak/planning/planner_registry.mojo)
- [browsecomp_plus_gold_faithfulness_frontier.mojo](../../benchmarks/browsecomp_plus_gold_faithfulness_frontier.mojo)
- [single_core_faithfulness_frontier.mojo](../../benchmarks/single_core_faithfulness_frontier.mojo)

Verified from the planner registry:

- `centroid_postings_imputed_flat` is `SEARCH_PLANNER_STATUS_PROMOTED`
- `centroid_postings_imputed` is `SEARCH_PLANNER_STATUS_REGRESSION_BASELINE`

That meant the previous frontier surface was benchmarking an older baseline
without benchmarking the promoted follow-on on the same lane.

## Change

Added `centroid_postings_imputed_flat_search_plan(...)` to:

- [browsecomp_plus_gold_faithfulness_frontier.mojo](../../benchmarks/browsecomp_plus_gold_faithfulness_frontier.mojo)
- [single_core_faithfulness_frontier.mojo](../../benchmarks/single_core_faithfulness_frontier.mojo)

Validation:

- `pixi run mojo -I . tests/test_faithfulness_frontier_json.mojo`

Observed:

- `2/2` passed

## Benchmark

Command run:

```bash
pixi run mojo -I . benchmarks/browsecomp_plus_gold_faithfulness_frontier.mojo
```

Artifact:

- `.cache/kayak/browsecomp_plus_gold_faithfulness_frontier.json`

## Measured result

Paired rows on `BrowseComp-Plus` gold:

- `candidate_k = 10`
  - `centroid_postings_imputed_flat`
    - `mean_search_seconds = 0.0010145`
    - `mean_ndcg@10 = 0.4449797343575519`
    - `mean_candidate_recall_at_final_k = 0.4`
  - `centroid_postings_imputed`
    - `mean_search_seconds = 0.0010275`
    - `mean_ndcg@10 = 0.4449797343575519`
    - `mean_candidate_recall_at_final_k = 0.4`
- `candidate_k = 20`
  - `centroid_postings_imputed_flat`
    - `mean_search_seconds = 0.00155`
    - `mean_ndcg@10 = 0.4234444809096049`
    - `mean_candidate_recall_at_final_k = 0.575`
  - `centroid_postings_imputed`
    - `mean_search_seconds = 0.00153025`
    - `mean_ndcg@10 = 0.4234444809096049`
    - `mean_candidate_recall_at_final_k = 0.575`
- `candidate_k = 40`
  - `centroid_postings_imputed_flat`
    - `mean_search_seconds = 0.00255575`
    - `mean_ndcg@10 = 0.39105250156773835`
    - `mean_candidate_recall_at_final_k = 0.75`
  - `centroid_postings_imputed`
    - `mean_search_seconds = 0.002666`
    - `mean_ndcg@10 = 0.39105250156773835`
    - `mean_candidate_recall_at_final_k = 0.75`
- `candidate_k = 80`
  - `centroid_postings_imputed_flat`
    - `mean_search_seconds = 0.00479025`
    - `mean_ndcg@10 = 0.31903977127133754`
    - `mean_candidate_recall_at_final_k = 0.975`
  - `centroid_postings_imputed`
    - `mean_search_seconds = 0.00486375`
    - `mean_ndcg@10 = 0.31903977127133754`
    - `mean_candidate_recall_at_final_k = 0.975`

Top judged-score approximate row on the current gold frontier:

- `centroid_postings_imputed_flat`, `candidate_k = 10`
  - `mean_search_seconds = 0.0010145`
  - `mean_ndcg@10 = 0.4449797343575519`

The previous top judged-score row on the same family was:

- `centroid_postings_imputed`, `candidate_k = 10`
  - `mean_search_seconds = 0.0010275`
  - `mean_ndcg@10 = 0.4449797343575519`

So the promoted flat-imputed row is faster by:

- `1.30e-05 s`

at the same measured judged quality on this slice.

## Interpretation

What is verified:

- the public gold frontier was missing a planner-promoted generator
- adding it changes the measured frontier slightly in Kayak's favor
- the gain is real but small on the current gold slice

What is not justified:

- claiming a new large engine breakthrough from this change alone
- claiming `centroid_postings_imputed_flat` dominates `centroid_postings_imputed`
  at every candidate window on this lane

The sound conclusion is narrower:

- the frontier benchmark is now more faithful to the engine's promoted search
  options
- the best judged-score approximate row on the current gold slice should be
  reported with `centroid_postings_imputed_flat`
