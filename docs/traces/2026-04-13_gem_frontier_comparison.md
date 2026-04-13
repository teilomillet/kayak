# GEM Frontier Comparison

Date: `2026-04-13`

## Objective

Measure the current in-tree `gem_graph` stage-1 path against the existing
stage-1 baselines on identical benchmark slices, while exposing graph-native
query-time counters instead of treating GEM as a black box.

This step was meant to answer a narrower question than "is GEM better?":

- can `kayak` now compare GEM honestly against the current stage-1 families
- and what does that comparison say on the current local benchmark surfaces

## Implementation Boundary

Added explicit graph-native counters to the planning/reporting path:

- [kayak/planning/graph_search_counters.mojo](../../kayak/planning/graph_search_counters.mojo)
- [kayak/planning/candidate_set.mojo](../../kayak/planning/candidate_set.mojo)
- [kayak/planning/stage_profile.mojo](../../kayak/planning/stage_profile.mojo)
- [kayak/planning/json.mojo](../../kayak/planning/json.mojo)
- [kayak/planning/execution_graph_family.mojo](../../kayak/planning/execution_graph_family.mojo)

The counters now recorded for graph-family stage 1 are:

- `visited_vertex_count`
- `expanded_edge_count`
- `visited_cluster_count`
- `entry_point_count`
- `max_frontier_size`

These counters are threaded through:

- explain output
- JSON explain serialization
- faithfulness frontier summaries

Added a reproducible GEM build helper for frontier comparisons:

- [kayak/benchmarks/gem_frontier_config.mojo](../../kayak/benchmarks/gem_frontier_config.mojo)

Current benchmark-facing heuristic:

- `fine_cluster_count = min(vector_dim, total_vector_count)`
- `coarse_cluster_count = min(fine_cluster_count, floor(sqrt(document_count)))`
- `cluster_cutoff = min(query_vector_budget, coarse_cluster_count)`

Reason:

- this is explicit and reproducible across slices
- it is not claimed to be paper-optimal GEM tuning
- it avoids inventing a benchmark-only hidden default inside the graph builder

Extended the benchmark mirror path so one collection snapshot can now carry:

- exact packed index
- document proxy
- centroid postings / heads
- GEM graph

Changed files:

- [kayak/collections/mirror.mojo](../../kayak/collections/mirror.mojo)
- [benchmarks/single_core_faithfulness_frontier.mojo](../../benchmarks/single_core_faithfulness_frontier.mojo)
- [benchmarks/browsecomp_plus_gold_faithfulness_frontier.mojo](../../benchmarks/browsecomp_plus_gold_faithfulness_frontier.mojo)

Important honesty boundary:

- this is **not** a true "WARP vs GEM" benchmark
- `centroid_postings_imputed` is the repo's closest current WARP-inspired
  baseline, not a faithful WARP reproduction

That boundary is already documented in:

- [docs/traces/2026-04-12_warp_gem_reading_and_imputed_stage.md](2026-04-12_warp_gem_reading_and_imputed_stage.md)

## Validation

Targeted tests rerun after the instrumentation:

```bash
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_service_json.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_service_contracts.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_faithfulness_frontier_json.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_collection_search_plan.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_collection_storage.mojo
```

Observed result:

- all listed tests passed locally

Benchmarks run:

```bash
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen bench_single_core_faithfulness_frontier_raw
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen bench_browsecomp_plus_gold_faithfulness_frontier_raw
```

Generated artifacts:

- `.cache/kayak/single_core_faithfulness_frontier.json`
- `.cache/kayak/browsecomp_plus_gold_faithfulness_frontier.json`

## Measured Result

### Synthetic Single-Core Frontier

Measured first full-recall point on the current stage-0 slices, using
`posting_cap = 0` for the non-head families:

- `docs_64`
  - `document_proxy`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000009`
  - `centroid_postings`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000016`
  - `centroid_postings_imputed`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000024`
  - `gem_graph`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000060`
- `docs_256`
  - `document_proxy`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000020`
  - `centroid_postings`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000019`
  - `centroid_postings_imputed`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000036`
  - `gem_graph`: full recall at `candidate_k = 8`, `mean_search_seconds ≈ 0.000083`
- `docs_1024`
  - `document_proxy`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000057`
  - `centroid_postings`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000029`
  - `centroid_postings_imputed`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000056`
  - `gem_graph`: no full-recall row in the measured sweep; observed recall stayed at `0.125`
- `docs_4096`
  - `document_proxy`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000211`
  - `centroid_postings`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000064`
  - `centroid_postings_imputed`: full recall at `candidate_k = 2`, `mean_search_seconds ≈ 0.000140`
  - `gem_graph`: no full-recall row in the measured sweep; observed recall stayed at `0.125`

Interpretation:

- on the smallest synthetic slice, current GEM works but is slower than the
  simpler baselines at the same recall point
- on the medium synthetic slice, current GEM reaches full recall only after a
  wider candidate window than the simpler baselines
- on the larger synthetic slices, the current GEM benchmark config is not
  competitive at all on recall

Representative graph counters at the first full-recall GEM row:

- `docs_64`, `candidate_k = 2`
  - `visited_vertex_count ≈ 49.5`
  - `expanded_edge_count ≈ 357.75`
  - `visited_cluster_count ≈ 2.75`
  - `entry_point_count ≈ 2.75`
  - `max_frontier_size ≈ 19.625`
- `docs_256`, `candidate_k = 8`
  - `visited_vertex_count ≈ 85.125`
  - `expanded_edge_count ≈ 658.75`
  - `visited_cluster_count ≈ 2.875`
  - `entry_point_count ≈ 1.875`
  - `max_frontier_size ≈ 23.75`

Those counters already suggest the current graph path is doing substantial work
relative to the corpus size even on the smaller synthetic slices.

### BrowseComp-Plus Gold Frontier

Measured first full-recall point on the current public gold slice:

- `exact_full_scan`: full recall at `candidate_k = 10`, `mean_search_seconds ≈ 0.005579`
- `document_proxy`: full recall at `candidate_k = 40`, `mean_search_seconds ≈ 0.003455`
- `centroid_postings`: full recall at `candidate_k = 90`, `mean_search_seconds ≈ 0.008897`
- `centroid_postings_imputed`: full recall at `candidate_k = 90`, `mean_search_seconds ≈ 0.009212`
- `gem_graph`: full recall at `candidate_k = 80`, `mean_search_seconds ≈ 0.008393`

At the more informative intermediate point `candidate_k = 40`:

- `document_proxy`
  - candidate recall `1.0`
  - `mean_search_seconds ≈ 0.003455`
- `centroid_postings`
  - candidate recall `0.8`
  - `mean_search_seconds ≈ 0.003712`
- `centroid_postings_imputed`
  - candidate recall `0.75`
  - `mean_search_seconds ≈ 0.004002`
- `gem_graph`
  - candidate recall `0.925`
  - `mean_search_seconds ≈ 0.004068`

Interpretation:

- current GEM is materially stronger than the current centroid-family baselines
  at the same `candidate_k` on this harder public slice
- current GEM is still not the best frontier point, because `document_proxy`
  already reaches full recall at `candidate_k = 40` while GEM still needs
  `candidate_k = 80`
- once GEM reaches full recall, it is still much slower than `document_proxy`
  on this slice and only slightly better than the centroid-family baselines

Representative graph counters at the first full-recall GEM row on this slice
(`candidate_k = 80`):

- `visited_vertex_count ≈ 88.75`
- `expanded_edge_count ≈ 536.5`
- `visited_cluster_count ≈ 7.75`
- `entry_point_count ≈ 5.75`
- `max_frontier_size ≈ 47.75`

This matters because the slice has `90` documents.

Inference:

- current GEM is almost touching the whole graph to recover full recall on this
  public slice
- that means the current graph path is not yet earning a real pruning
  advantage there

## Conclusion

This step succeeded technically and was useful epistemically.

What is now true:

- `kayak` can compare the current GEM path against the existing stage-1
  baselines using one shared frontier format
- the benchmark artifacts now expose graph-native query-time counters
- the current GEM path is stronger than the current centroid-family baselines
  on BrowseComp gold at equal `candidate_k`
- the current GEM path is **not** the best measured frontier point on the
  current local surfaces

What is not justified:

- claiming that GEM currently beats the in-tree baselines overall
- claiming that the current graph build defaults are good enough
- claiming that Kayak now has a winning native graph frontier

The measured decision is narrower:

- keep the instrumentation
- keep the comparison harness
- do **not** promote `gem_graph` as the preferred default stage-1 path yet
- next work should focus on graph construction and query-time pruning quality,
  because the counters show the current path is exploring too much graph to
  earn its recall

