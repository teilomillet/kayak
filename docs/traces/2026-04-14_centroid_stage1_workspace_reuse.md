# Centroid Stage-1 Workspace Reuse

## Claim

The centroid-family stage-1 paths still paid an avoidable per-query
`O(document_count)` setup tax by rebuilding dense score state for every query.
Because the execution seam already treats untouched documents through one
uniform `inactive_score`, the sound next step was:

- keep planner and candidate-generation policy explicit
- add an explicit reusable centroid workspace for repeated searches
- make the segment score result sparse over touched documents

The implementation should preserve search behavior while removing dense
per-query score initialization from the reusable path.

## Why this design

I inspected the current execution seam before editing:

- `CentroidSegmentScoreResult` is only consumed inside
  `kayak/planning/execution_centroid_family.mojo`
- inactive documents are already handled there through
  `score_result.inactive_score`
- only active documents need explicit materialized scores for candidate
  insertion

That meant a hidden global cache was not necessary and would have violated the
repo's preference for explicit behavior. The cleaner design was:

- `CentroidSegmentScoreResult` becomes sparse over active docs
- a new explicit `MutableCentroidCandidateGenerationWorkspace` owns reusable
  centroid scratch across repeated searches
- the convenience APIs still exist and allocate a fresh workspace when the
  caller does not provide one

This keeps policy visible while making the reusable hot path allocation-light.

## Implementation

Changed result contract:

- `kayak/planning/centroid_segment_score_result.mojo`
  - `CentroidSegmentScoreResult` now stores:
    - `active_scores`
    - `active_doc_indices`
    - `inactive_score`
  - added `materialize_centroid_segment_scores(...)` for the dense wrapper
    paths still used by tests and helper APIs

Added reusable sparse accumulation:

- `kayak/planning/centroid_primitives.mojo`
  - `MutableCentroidSelectionScratch` can now grow with
    `ensure_document_count(...)`
  - added `MutableCentroidSegmentAccumulator`
  - added `accumulate_selected_centroid_scores_with_accumulator(...)`

Added explicit centroid-family workspace:

- `kayak/planning/centroid_candidate_generation_workspace.mojo`
  - new `MutableCentroidCandidateGenerationWorkspace`

Threaded the reusable path through centroid-family stage-1 execution:

- `kayak/planning/execution.mojo`
  - added `candidate_generation_for_plan_with_workspace(...)`
  - added `search_collection_for_plan_with_workspace(...)`
- `kayak/planning/execution_centroid_family.mojo`
  - added `candidate_generation_for_centroid_family_with_workspace(...)`
  - candidate insertion now consumes sparse active scores directly

Threaded workspace-aware stage implementations:

- `kayak/planning/centroid_postings_stage.mojo`
- `kayak/planning/centroid_postings_flat_stage.mojo`
- `kayak/planning/centroid_postings_head_stage.mojo`
- `kayak/planning/centroid_postings_head_auto_stage.mojo`
- `kayak/planning/centroid_postings_blockmax_stage.mojo`
- `kayak/planning/centroid_postings_imputed_stage.mojo`
- `kayak/planning/centroid_postings_imputed_flat_stage.mojo`

Updated benchmark surfaces that naturally reuse a loaded snapshot:

- `benchmarks/profile_stage1_blockmax_real_subset.mojo`
- `kayak/benchmarks/stage_aware_json.mojo`
- added focused A/B benchmark:
  - `benchmarks/profile_centroid_workspace_reuse.mojo`

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_stage_aware_benchmark_json.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . tests/test_imputed_high_centroid_probe.mojo
pixi run mojo -I . benchmarks/profile_centroid_workspace_reuse.mojo
```

Observed correctness results:

- `tests/test_collection_search_plan.mojo`: `36/36` passed
- `tests/test_stage_aware_benchmark_json.mojo`: `4/4` passed
- `tests/test_service_runtime.mojo`: `26/26` passed
- `tests/test_imputed_high_centroid_probe.mojo`: `2/2` passed

Focused benchmark artifact:

- wrote `.cache/kayak/profile_centroid_workspace_reuse.tsv`

## Benchmark result

The focused benchmark compared:

- `fresh_workspace`: existing convenience path that constructs a fresh centroid
  workspace per query
- `reused_workspace`: the new explicit workspace path reused across the same
  SciFact query loop and snapshot

Measured means on `SciFact / scifact_real_subset / candidate_k=40`:

- `centroid_postings`
  - fresh: `8.987204181586203e-05 s`
  - reused: `9.465419048364785e-05 s`
  - ratio: `1.053211`
- `centroid_postings_flat`
  - fresh: `8.69277145771729e-05 s`
  - reused: `8.539511390163406e-05 s`
  - ratio: `0.982369`
- `centroid_postings_imputed`
  - fresh: `4.246528079429103e-04 s`
  - reused: `4.3699111086263766e-04 s`
  - ratio: `1.029055`
- `centroid_postings_imputed_flat`
  - fresh: `4.6828080808080806e-04 s`
  - reused: `5.348612012987013e-04 s`
  - ratio: `1.142180`

## Interpretation

What is verified:

- the explicit reusable centroid workspace is implemented end to end
- the centroid-family execution seam now supports sparse active-doc score
  results without changing candidate behavior
- repeated-search benchmark surfaces can opt into workspace reuse explicitly
- correctness still holds across plain, flat, head, blockmax, imputed, and
  filtered centroid paths

What is **not** verified:

- a general latency win from the new workspace path on current public slices

The first focused SciFact A/B run is mixed:

- small win for `centroid_postings_flat`
- regressions or ambiguity for the other measured variants

So the sound current conclusion is:

- the API and execution change is correct
- the current public-slice benchmark does **not** yet justify claiming a broad
  performance improvement from workspace reuse alone

## Open next step

The next useful check is not more code speculation. It is one larger-doc-count
focused benchmark, preferably on a heavier public slice or a controlled
single-segment synthetic scale, to test whether the removed `O(document_count)`
setup tax becomes visible once stage-1 selection work no longer dominates the
same query loop.
