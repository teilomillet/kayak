# Blockmax Token Scratch Reuse

## Claim

`centroid_postings_blockmax` was still rebuilding dense token-local state on
every query token. Reusing generation-stamped scratch for token best scores and
touched docs should reduce stage-1 latency, but the top-`k` membership path
must stay safe for larger `candidate_k` windows because the repo benchmarks
exercise shortlist budgets well beyond `40`.

## Why this change

- `kayak/planning/centroid_postings_blockmax_stage.mojo` rebuilt per-token
  best-score and seen-flag state even after the exact/head stages had already
  moved to reusable scratch.
- the same file also kept a dense `top_positions` map for every document, even
  though the public stage-1 profile and most shortlist regimes use small
  `candidate_k`
- direct repo inspection showed the larger-budget risk is real:
  - `benchmarks/real_subset_candidate_window_sweep.mojo` sweeps up to
    `final_k * 8` and then full-document windows
  - `benchmarks/hard_recall_real_subset.mojo` explicitly evaluates
    `candidate_k = 100` and `candidate_k = 1000`

That means an optimization that only looks good at `candidate_k = 40` is not
sound enough on its own.

## Implementation

- added `MutableCentroidPostingBlockmaxScratch` in
  `kayak/planning/centroid_postings_blockmax_stage.mojo`
  - reuses `MutableCentroidSelectionScratch` for token best scores and touched
    docs
  - keeps a blockmax-specific top-position table only when
    `candidate_k > 64`
- removed the repeated per-token dense setup for:
  - token best scores
  - token seen flags
  - token active-doc tracking
- replaced the unconditional dense top-position bookkeeping with an adaptive
  policy:
  - small shortlist windows scan the local top-`k` list directly
  - larger shortlist windows use a generation-stamped dense position table
- added a focused kernel probe benchmark:
  - `benchmarks/profile_blockmax_candidate_k_probe.mojo`
  - measures direct `centroid_postings_blockmax` kernel latency across multiple
    `candidate_k` values on public real-subset slices

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . benchmarks/profile_stage1_blockmax_real_subset.mojo
pixi run mojo -I . benchmarks/profile_blockmax_candidate_k_probe.mojo
```

Observed correctness results:

- `tests/test_collection_search_plan.mojo`: `34/34` passed
- `tests/test_service_runtime.mojo`: `26/26` passed

### Exact stage-1 profile

Fresh baseline measured earlier in this session before the blockmax scratch
rewrite, on the same SciFact slice and benchmark harness:

- `centroid_postings_blockmax`: `0.00020923413541932374 s`

Measured after the final kept implementation:

- `centroid_postings_blockmax`: `0.0001803625238910815 s`

Derived speedup:

- about `1.16x`

### Direct `candidate_k` probe

The probe was run once before the adaptive/lazy top-position rewrite and once
after it.

Before the final adaptive/lazy version:

- `SciFact`, `candidate_k = 40`: `8.31688773739277e-05 s`
- `SciFact`, `candidate_k = 55`: `8.335079914524574e-05 s`
- `BrowseComp-Plus`, `candidate_k = 40`: `9.007682108668558e-05 s`
- `BrowseComp-Plus`, `candidate_k = 80`: `9.21384204307428e-05 s`
- `BrowseComp-Plus`, `candidate_k = 90`: `9.25887396085843e-05 s`

After the final adaptive/lazy version:

- `SciFact`, `candidate_k = 40`: `8.26746887763837e-05 s`
- `SciFact`, `candidate_k = 55`: `8.318686207152213e-05 s`
- `BrowseComp-Plus`, `candidate_k = 40`: `9.056842678135287e-05 s`
- `BrowseComp-Plus`, `candidate_k = 80`: `9.178354306042264e-05 s`
- `BrowseComp-Plus`, `candidate_k = 90`: `9.230276588079895e-05 s`

Observed deltas versus the pre-adaptive probe:

- `SciFact`, `candidate_k = 40`: about `1.006x` faster
- `SciFact`, `candidate_k = 55`: about `1.002x` faster
- `BrowseComp-Plus`, `candidate_k = 40`: about `0.55%` slower
- `BrowseComp-Plus`, `candidate_k = 80`: about `1.004x` faster
- `BrowseComp-Plus`, `candidate_k = 90`: about `1.003x` faster

## Interpretation

This is a verified stage-1 improvement for the current `blockmax` path. The
main exact profile improved by about `1.16x`, and the focused probe shows the
kept implementation is neutral-to-better on the measured shortlist windows that
cross the adaptive threshold.

The public probe still does not validate truly large-window behavior on a slice
with `document_count >= 1000`; the public subsets cap below that. What is
verified today is:

- the repo does exercise larger shortlist budgets in its benchmark entrypoints
- the final implementation no longer makes linear top-`k` membership checks the
  only path for those budgets
- the small-window public slices did not regress materially

The next sound verification step, if we want to push this path further, is a
dedicated larger-slice or synthetic blockmax probe that reaches the `100` and
`1000` shortlist budgets already present in the hard-recall benchmark suite.
