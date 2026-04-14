# Head Token Scratch Reuse

## Claim

`centroid_postings_head` and `centroid_postings_head_auto` were paying repeated
`O(document_count)` setup per query token by rebuilding token-local best-score
and seen-flag arrays. Reusing generation-stamped scratch across tokens should
reduce stage-1 latency without changing shortlist behavior.

## Why this change

- `kayak/planning/centroid_postings_head_stage.mojo` rebuilt
  `token_best_scores` and `token_active_flags` for every query token.
- `kayak/planning/centroid_postings_head_auto_stage.mojo` did the same.
- The exact/imputed centroid path already had a proven reusable scratch
  structure in `MutableCentroidSelectionScratch`, so reusing that pattern was
  lower risk than introducing a new scratch representation.

## Implementation

- `centroid_postings_head_stage.mojo` now creates one
  `MutableCentroidSelectionScratch` per segment and reuses it across query
  tokens.
- `centroid_postings_head_auto_stage.mojo` now does the same.
- The token-local touched-doc tracking now uses generation stamps instead of a
  freshly zeroed token-active bitmap per token.

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . benchmarks/profile_stage1_blockmax_real_subset.mojo
```

Observed correctness results:

- `tests/test_collection_search_plan.mojo`: `34/34` passed
- `tests/test_service_runtime.mojo`: `25/25` passed

Benchmark comparison:

Fresh baseline on current `main` before the edit, SciFact slice,
`candidate_k=40`:

- `centroid_postings_head`: `0.00018456073939210034 s`
- `centroid_postings_head_auto`: `0.00018471005723218114 s`

Measured after the scratch reuse change on the same benchmark slice:

- `centroid_postings_head`: `0.0001619090237348858 s`
- `centroid_postings_head_auto`: `0.00016513041874192952 s`

Derived speedups:

- `centroid_postings_head`: about `1.14x`
- `centroid_postings_head_auto`: about `1.12x`

## Interpretation

This is a verified stage-1 win, not just a primitive-level improvement. The
benefit is moderate rather than dramatic, which matches the code shape: this
change removes repeated dense token-local setup, but it does not yet address the
remaining dense score materialization or the blockmax-specific per-token state.
