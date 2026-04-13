# 2026-04-13: explicit reclaim executor for inactive snapshots

## Why this step

The previous reclaim tranche added a collection-level planner, but not an
executor. That was intentional: deletion without a first-class plan would have
been too easy to get wrong.

With the planner in place, the next sound step was an executor that:

- consumes an explicit `CollectionReclaimPlan`
- supports dry-run by default
- refuses to act on stale plan state
- deletes in a monotonic-safe order

## What changed

Updated files:

- [kayak/collections/reclaim.mojo](../../kayak/collections/reclaim.mojo)
- [kayak/collections/reclaim_runtime.mojo](../../kayak/collections/reclaim_runtime.mojo)
- [kayak/collections/reclaim_json.mojo](../../kayak/collections/reclaim_json.mojo)
- [kayak/collections/__init__.mojo](../../kayak/collections/__init__.mojo)
- [kayak/__init__.mojo](../../kayak/__init__.mojo)
- [tests/test_collection_reclaim.mojo](../../tests/test_collection_reclaim.mojo)
- [tests/test_collection_reclaim_json.mojo](../../tests/test_collection_reclaim_json.mojo)

New contract:

- `CollectionReclaimExecutionResult`

New runtime entry point:

- `execute_collection_reclaim_plan(...)`

New JSON surface:

- `collection_reclaim_execution_result_json(...)`

## Execution semantics

The executor validates the plan against current on-disk state before doing
anything:

- collection ids must still match
- active snapshot id must still match
- snapshot count and per-snapshot generation/size facts must still match
- reclaimable unique segment ids and bytes must still match

If those checks fail, execution aborts as a stale-plan rejection.

If execution proceeds:

1. reclaimable inactive snapshot roots are removed first
2. remaining retained snapshots are re-read
3. only then are unique inactive-only segment roots removed

That ordering is deliberate.

It is not transactional, but it is monotonic-safe:

- removing inactive snapshot roots first cannot break retained search state
- segment deletion happens only after the retained snapshot reachability set is
  recomputed

## Guardrails

What is now justified:

- `kayak` can execute a previously built reclaim plan
- dry-run leaves storage untouched and still returns a concrete execution result
- apply mode removes only the inactive snapshots and inactive-only segments the
  plan proved reclaimable
- stale plans are rejected before deletion starts

What is not justified:

- background or automatic cleanup
- transactional multi-root deletion
- persisted retention policy on collection or snapshot manifests
- service-level orchestration or scheduling

## Verification

Focused suites rerun after the change:

- `pixi run test_collection_reclaim`
- `pixi run test_collection_reclaim_json`
- `pixi run test_service_runtime`

All passed on this branch.

## Conclusion

This gives the hosted engine an explicit cleanup primitive without hiding policy
or reachability behind heuristics.

The next sound step would be to surface this through service/report contracts or
to add persisted retention metadata, but only after deciding which policy model
the hosted engine should actually own.
