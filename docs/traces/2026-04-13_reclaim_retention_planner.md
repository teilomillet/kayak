# 2026-04-13: reclaim and retention planner for inactive snapshots

## Why this step

After the hosted engine gained:

- explicit active-snapshot publication
- compacted draft baselines
- operational metrics for inactive snapshots and pending drafts

the next substrate gap was deletion policy.

It would have been unsound to jump directly to cleanup, because the repository
did not yet have a first-class way to answer:

- which inactive snapshots are intentionally retained
- which inactive snapshots are reclaim candidates
- which physical segments are uniquely reclaimable under a given retention
  policy

This step adds that planning layer without deleting anything.

## What changed

New files:

- [kayak/collections/reclaim.mojo](../../kayak/collections/reclaim.mojo)
- [kayak/collections/reclaim_runtime.mojo](../../kayak/collections/reclaim_runtime.mojo)
- [kayak/collections/reclaim_json.mojo](../../kayak/collections/reclaim_json.mojo)
- [tests/test_collection_reclaim.mojo](../../tests/test_collection_reclaim.mojo)
- [tests/test_collection_reclaim_json.mojo](../../tests/test_collection_reclaim_json.mojo)

Updated exports:

- [kayak/collections/__init__.mojo](../../kayak/collections/__init__.mojo)
- [kayak/__init__.mojo](../../kayak/__init__.mojo)

Updated task surface:

- [pyproject.toml](../../pyproject.toml)

## Planner contract

The planner is collection-scoped and policy-explicit.

Policy:

- `keep_latest_inactive_count`
- `pinned_snapshot_ids`

Output:

- `CollectionReclaimPlan`
- per-snapshot `SnapshotRetentionDecision`
- aggregate reclaimable unique segment ids and bytes

Important semantics:

- the active snapshot is always retained
- pinned snapshots are retained explicitly
- the remaining inactive snapshots are retained or reclaimed according to the
  caller-provided count policy
- reclaimable physical segments are counted only if they are not referenced by
  the active snapshot or by any retained inactive snapshot

This means the planner reports physical reclaim candidates, not just logical
inactive snapshots.

## Guardrails

What is now justified:

- `kayak` can build a non-destructive collection-level reclaim plan directly
  from stored manifests
- the plan can distinguish retained versus reclaimable inactive snapshots
- the plan can identify unique inactive-only segment roots and sum their bytes
- explicit pinning and "keep latest inactive N" policies both work

What is not justified:

- any automatic deletion or garbage collection
- any persisted retention metadata on snapshots themselves
- any service-wide shared-storage reclaim accounting across collections

The planner is intentionally narrow:

- it does not mutate manifests
- it does not delete files
- it does not guess a default retention policy beyond what the caller asks for

## Verification

Focused suites rerun after the change:

- `pixi run test_collection_reclaim`
- `pixi run test_collection_reclaim_json`
- `pixi run test_service_runtime`

All passed on this branch.

## Conclusion

This is the right step before reclaim execution:

- policy is explicit instead of implicit
- reclaimable bytes are derived from real manifest reachability
- future cleanup work can consume a plan rather than re-deriving retention logic
  ad hoc
