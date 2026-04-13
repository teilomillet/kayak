## Goal

Factor centroid-family execution behavior into an explicit primitive contract so
the hot runtime path no longer repeats a long list of centroid variant names for
artifact requirements, weight-order assumptions, and shortlist-budget behavior.

## Why this tranche

Before this change, `execution_centroid_family.mojo` directly encoded:

- which centroid variants use `centroid_heads` vs `centroid_postings`
- which variants require weight-sorted postings
- which variants consume `candidate_k` vs `final_k` as a shortlist budget
- which scoring kernel should run for each variant

That worked, but it scattered execution semantics across a large branch block in
the search path. It made the code harder to browse and increased the risk that a
new centroid variant would update one place but not another.

## Changes

- Added `kayak/planning/centroid_execution_contract.mojo`
  - defines `CentroidExecutionContract`
  - maps each centroid generator kind to:
    - artifact family
    - score variant
    - weight-sorted-postings requirement
    - shortlist-budget source
    - whether the full query object is consumed
- Updated `kayak/planning/execution_centroid_family.mojo`
  - uses the contract for artifact loading and validation
  - uses the contract for weight-sorted postings checks
  - drives kernel selection through contract metadata instead of repeating
    variant-specific string checks throughout the execution loop
- Exported the new contract through `kayak/planning/__init__.mojo` and
  `kayak/__init__.mojo`
- Added `tests/test_centroid_execution_contract.mojo`

## Verification

Executed:

- `pixi run mojo -I . tests/test_centroid_execution_contract.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_service_runtime.mojo`

Observed:

- the new direct contract tests passed
- the full collection-search-plan suite passed, including centroid postings,
  flat, head, head-auto, blockmax, imputed, and imputed-flat paths
- the hosted runtime suite passed after the refactor, covering planned search,
  native-goal execution, exact-filter guardrails, gem-graph configuration, and
  lifecycle flows

## Outcome

Kayak now has a clearer primitive for centroid execution semantics. The runtime
still supports the same centroid family behaviors, but the operational meaning
of each variant is centralized in one contract instead of being duplicated in
the hot search path.
