# 2026-04-13: operational metrics for inactive snapshots and pending drafts

## Why this step

After active-snapshot publication and draft-baseline compaction were in place,
the next missing piece was situational awareness:

- what is live right now
- what is stored but inactive
- what unpublished draft work is still pending

Those questions are substrate questions, not stage-1 questions. They matter for
hosted continuity and storage control regardless of whether the long-term
winner is centroid, GEM, WARP-like, or something else.

## What changed

Updated files:

- [kayak/service/service_status.mojo](../../kayak/service/service_status.mojo)
- [kayak/service/metrics_runtime.mojo](../../kayak/service/metrics_runtime.mojo)
- [kayak/service/json.mojo](../../kayak/service/json.mojo)
- [tests/test_service_contracts.mojo](../../tests/test_service_contracts.mojo)
- [tests/test_service_json.mojo](../../tests/test_service_json.mojo)
- [tests/test_service_runtime.mojo](../../tests/test_service_runtime.mojo)

`ServiceMetricsSnapshot` now exposes additional counters for:

- `published_snapshot_count`
- `inactive_snapshot_count`
- `inactive_unique_segment_count`
- `inactive_unique_byte_size`
- `pending_draft_collection_count`
- `pending_draft_mutation_count`

The runtime computes those counters directly from stored manifests:

- live state still comes only from the active snapshot
- inactive snapshot counts come from stored snapshot manifests under the same
  collection
- inactive unique segment bytes are counted only for segment roots not
  referenced by the active snapshot
- pending draft counters come from draft-state metadata, not by reloading the
  full draft payload

## Guardrails

What is now justified:

- the service can report how many stored snapshots are not currently active
- the service can report the unique byte size of inactive-only segments
- the service can report whether collections still have unpublished draft
  mutations pending

What is not justified:

- any automatic reclaim policy for inactive snapshots or segments
- any claim that inactive unique byte size equals reclaimable bytes under every
  future retention policy
- any filter-aware or approximate stage-1 improvement

The semantics are intentionally narrow:

- counts are derived from the current active snapshot boundary
- inactive unique bytes are physical-segment-root bytes, not logical snapshot
  bytes
- pending drafts mean mutation-log entries still exist after the last publish

## Verification

Focused suites rerun after the change:

- `pixi run test_service_contracts`
- `pixi run test_service_json`
- `pixi run test_service_runtime`

All passed on this branch.

## Conclusion

This tranche gives `kayak` better hosted-engine visibility without choosing a
stage-1 architecture:

- live versus inactive state is explicit
- pending unpublished work is visible
- the next step can build reclaim or retention policy on measured facts rather
  than guesses
