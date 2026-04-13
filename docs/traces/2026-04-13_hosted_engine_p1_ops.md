# 2026-04-13: hosted-engine P1 operational tranche

## Why this step

After the hosted-engine P0 tranche landed, two operational gaps were still
blocking real engine use:

- there was no runtime path that aggregated service-level metrics from the
  published manifests already on disk
- compaction existed only as a contract, not as an executable snapshot-safe
  replacement flow

This step addresses those gaps on the current engine boundary without making
any claim about broader late-interaction asymptotics.

## What changed

### Service metrics from published state

New file:

- [kayak/service/metrics_runtime.mojo](../../kayak/service/metrics_runtime.mojo)

Exports:

- `build_service_health_status(...)`
- `build_service_metrics_snapshot(...)`

What is now explicit:

- service collection discovery under the hosted layout
- visible snapshot counting from collection manifests plus matching live
  snapshot generations
- aggregate segment, document, vector, and byte counts over the published
  collection state

Important epistemic constraint:

- the implementation is intentionally strict
- if a collection claims a live generation but no unique matching snapshot
  exists, the runtime raises instead of silently guessing

### Live-snapshot compaction and replacement

New file:

- [kayak/collections/compaction_runtime.mojo](../../kayak/collections/compaction_runtime.mojo)

Exports:

- `build_compaction_plan_for_snapshot(...)`
- `execute_compaction_plan(...)`

What is now explicit:

- compaction planning against a specific live snapshot
- source-segment verification against the published snapshot
- sealing a new merged segment
- publishing a replacement snapshot with a new generation
- preserving the old snapshot instead of mutating visible state in place

Important limit:

- this is a live-snapshot-only compaction path
- it does not yet reclaim old segments
- it does not yet provide atomic multi-file publish semantics across crash
  boundaries

## Guardrails

Tests rerun after the change:

- `pixi run test_collection_compaction`
- `pixi run test_service_runtime`
- `pixi run test_collection_resolution`
- `pixi run test_collection_search_plan`

All passed on this branch.

New coverage added:

- service-level metrics aggregate the visible snapshots of multiple hosted
  collections
- compaction republishes a replacement snapshot and keeps the old snapshot
  intact

## Verified implementation claims

What is now justified:

- `kayak` can compute service-level health and storage counters directly from
  the published manifest tree
- `kayak` can compact the live published snapshot into a new sealed segment and
  publish a replacement snapshot with a new generation

What is not justified by this step:

- background maintenance scheduling
- compaction backlog metrics
- automatic cleanup of superseded segments
- atomic commit/rollback semantics for snapshot publication
- any retrieval-quality improvement claim

## Conclusion

This tranche makes the hosted engine more operationally credible:

- the service can report what is actually published
- compaction is no longer just a plan object

The next sound steps remain:

- metadata sidecars and staged filters with explicit semantics
- richer operational reporting and JSON
- stronger publication atomicity
- heavier native stage-1 work only after these boundaries stay stable
