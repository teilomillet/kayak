# 2026-04-13: exact-only doc_id filter capability

## Why this step

The hosted runtime had a blanket `match_all` requirement even though the filter
contract already existed. That made the API look more complete than the engine
really was.

The goal of this step was not to claim full filter support. It was to replace
the blanket rejection with a narrow, explicit capability boundary that the
engine can actually honor today.

## What changed

New file:

- [kayak/filters/runtime.mojo](../../kayak/filters/runtime.mojo)

Updated files:

- [kayak/service/runtime.mojo](../../kayak/service/runtime.mojo)
- [kayak/planning/execution.mojo](../../kayak/planning/execution.mojo)
- [kayak/planning/execution_exact_family.mojo](../../kayak/planning/execution_exact_family.mojo)
- [kayak/planning/exact_stage.mojo](../../kayak/planning/exact_stage.mojo)
- [kayak/planning/explain.mojo](../../kayak/planning/explain.mojo)

What is now explicit:

- `match_all` remains supported
- exact-only `doc_id` filters are now supported
- non-`doc_id` filters are still rejected
- non-exact stage-1 plans are still rejected for filtered search

The runtime now enforces a capability progression instead of pretending all
filters are equivalent:

- `match_all`
- exact-only `doc_id`
- future metadata-sidecar filters
- future candidate-pushdown filters

## Guardrails

Tests rerun after the change:

- `pixi run test_filters`
- `pixi run test_service_runtime`
- `pixi run test_collection_search_plan`

All passed on this branch.

New assertions added:

- runtime helper coverage for exact `doc_id` filter detection and matching
- hosted runtime search accepts exact `doc_id` filtering
- hosted runtime still rejects filtered non-exact stage-1 plans

## Verified implementation claim

What is justified:

- the hosted runtime supports a narrow exact-only filter capability for
  `doc_id`

What is not justified:

- metadata filtering
- filtered approximate stage-1 retrieval
- sidecar-based filter pushdown
- complete filter support

## Conclusion

This step makes the service contract more honest and more useful:

- users can now perform one real filtered search mode
- the engine still explicitly rejects the wider cases it cannot yet honor

That is the correct substrate move before metadata-sidecar design, not after.
