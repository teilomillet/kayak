# Imputed Selection High-Centroid Bound

## Claim

The imputed selectors were keeping a fully sorted list across all centroids even
though they only ever consume the top `bound`, where
`bound = min(centroid_count, 128)`. Replacing the full sort with bounded
top-`bound` insertion should help once `centroid_count` exceeds `128`, but that
claim is only meaningful on a slice that actually reaches that regime.

## Why this step

An earlier version of this optimization was intentionally backed out because the
repo's measured public slices and default synthetic fixtures stayed at or below
the current imputed bound. On those slices the optimization is structurally
inert: `bound == centroid_count`, so there is no tail work to trim.

This step closes that evidence gap by adding a deterministic benchmark-only
synthetic fixture that produces:

- `vector_dim = 180`
- `centroid_count = 180`

That is enough to test the selector behavior above the current `128`-centroid
bound without changing planner or stage contracts.

## Implementation

### New high-centroid fixture seam

- `kayak/benchmarks/synthetic_hard_recall_fixture.mojo`
  - added `high_centroid_synthetic_hard_recall_profile()`
- `kayak/benchmarks/__init__.mojo`
  - exports the new profile helper

The profile is benchmark-only and intentionally narrow:

- `slot_count = 4`
- `values_per_slot = 4`
- `filler_vector_count = 32`
- `filler_concept_pool_size = 160`
- `duplicates_per_combination = 2`
- `query_count = 8`
- `final_k = 2`

### New benchmark and guardrail

- `benchmarks/profile_imputed_selection_high_centroid.mojo`
  - mirrors the synthetic fixture with a full centroid budget
  - asserts `centroid_count > 128`
  - benchmarks:
    - `imputed_selection_nested`
    - `imputed_selection_flat`
  - writes `.cache/kayak/profile_imputed_selection_high_centroid.tsv`
- `tests/test_imputed_high_centroid_probe.mojo`
  - asserts the mirrored centroid postings index exceeds the current bound
  - keeps the stage-level semantic contract:
    `centroid_postings_imputed_flat` must match `centroid_postings_imputed`
    scores on this same high-centroid slice

### Selector change

- `kayak/planning/centroid_postings_imputed_stage.mojo`
- `kayak/planning/centroid_postings_imputed_flat_stage.mojo`

Both selectors now keep only the ordered top `bound` centroids during scoring
instead of inserting all centroids into a fully sorted list and truncating
later.

That keeps the observed semantics intact because downstream logic only reads:

- the top `nprobe <= bound` centroids
- the top `bound` prefix for the `t'` / baseline-correction walk

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_imputed_high_centroid_probe.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . benchmarks/profile_imputed_selection_high_centroid.mojo
```

Observed correctness results:

- `tests/test_imputed_high_centroid_probe.mojo`: `2/2` passed
- `tests/test_collection_search_plan.mojo`: `34/34` passed
- `tests/test_service_runtime.mojo`: `26/26` passed

Observed high-centroid benchmark regime:

- `slice_name = imputed_probe_centroids180_docs544`
- `vector_dim = 180`
- `centroid_count = 180`

## Measured outcome

Fresh baseline on the new high-centroid probe before the selector change:

- `imputed_selection_nested`: `0.00011665367511117775 s`
- `imputed_selection_flat`: `0.00020167996559664845 s`

Measured after the bounded-top-`bound` selector rewrite:

- `imputed_selection_nested`: `0.00011442234428542634 s`
- `imputed_selection_flat`: `0.00019968965996608001 s`

Derived speedups:

- `imputed_selection_nested`: about `1.020x`
- `imputed_selection_flat`: about `1.010x`

## Interpretation

This is a real but modest selector win. The important part is not the absolute
size of the improvement; it is that the repo now has a benchmark seam where the
claim is mechanically testable.

What is now verified:

- the repo can benchmark imputed selection above the current `128`-centroid
  bound
- the bounded selector keeps the current nested/flat semantic contract on that
  slice
- the bounded selector is measurably faster on that high-centroid workload

What is still not claimed:

- that this change materially shifts the public frontier on today's default
  public slices
- that this alone solves the larger imputed-selection hotspot identified in the
  earlier primitive trace

The correct next interpretation is narrower:

- the earlier optimization idea is now validated on the regime where it should
  matter
- the remaining imputed-selection work should target larger algorithmic costs
  than full-tail sorting alone
