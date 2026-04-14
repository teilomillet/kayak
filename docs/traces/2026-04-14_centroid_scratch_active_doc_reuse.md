# Centroid Scratch Active-Doc Reuse

## Claim

`MutableCentroidSelectionScratch` still allocated a fresh touched-doc list on
every query token, even after the earlier generation-stamped scratch work. If
that list is reused with an explicit active-count cursor, the exact accumulation
primitive should get cheaper and that gain should surface in at least the plain
`centroid_postings` stage-1 path.

## Why this change

- `kayak/planning/centroid_primitives.mojo` reset token-local touched docs with
  `scratch.token_active_doc_indices = List[Int]()`
- the same scratch is reused by:
  - `centroid_postings`
  - `centroid_postings_head`
  - `centroid_postings_head_auto`
  - `centroid_postings_blockmax`
- that meant a per-token allocation remained in a shared hot primitive even
  after the earlier score/seen-buffer reuse changes

The sound next step was therefore to remove that shared allocation once, at the
primitive seam, and then validate both the primitive and a real stage-1 path.

## Implementation

- `MutableCentroidSelectionScratch` now owns:
  - `token_active_doc_indices`
  - `token_active_doc_count`
- `begin_token()` now resets only the active-count cursor instead of allocating
  a brand-new list
- added `append_token_active_doc_index(...)` so touched-doc writes reuse the
  existing buffer when capacity is already present
- updated all current scratch users to iterate only up to
  `token_active_doc_count`:
  - `kayak/planning/centroid_primitives.mojo`
  - `kayak/planning/centroid_postings_head_stage.mojo`
  - `kayak/planning/centroid_postings_head_auto_stage.mojo`
  - `kayak/planning/centroid_postings_blockmax_stage.mojo`

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . benchmarks/profile_centroid_primitives_real_subset.mojo
pixi run mojo -I . benchmarks/profile_stage1_blockmax_real_subset.mojo
```

Observed correctness results:

- `tests/test_collection_search_plan.mojo`: `34/34` passed
- `tests/test_service_runtime.mojo`: `26/26` passed

### Primitive benchmark

Fresh baseline measured earlier in this session before the active-doc reuse
rewrite, on the same SciFact primitive harness:

- `exact_accumulation`: `3.832447698368441e-05 s`

Measured after the rewrite:

- `exact_accumulation`: `3.181489759953803e-05 s`

Derived speedup:

- about `1.20x`

### Stage-1 benchmark

Fresh baseline measured earlier in this session before the rewrite, on the same
SciFact stage-1 harness:

- `centroid_postings`: `9.404405280292312e-05 s`

Measured after the rewrite:

- `centroid_postings`: `8.713329050494667e-05 s`

Derived speedup:

- about `1.08x`

## Interpretation

This is a real primitive-to-stage payoff rather than a microbenchmark-only
artifact.

What is verified:

- the shared touched-doc buffer no longer reallocates every token
- the exact accumulation primitive is materially faster on the measured public
  slice
- the plain `centroid_postings` stage-1 path also improves on the same host and
  benchmark family

What is not yet claimed:

- a precise stage-level speedup for every scratch consumer
- a large win for the imputed family specifically

Those still need direct reruns if we want to promote broader claims. But this
change clears the bar for keeping the shared primitive rewrite: it is simpler
than per-stage workarounds and is now backed by measured evidence.
