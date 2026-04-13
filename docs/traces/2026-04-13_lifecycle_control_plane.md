# Trace: Lifecycle Control Plane

Date: `2026-04-13`

## Goal

Finish the hosted-engine control-plane tranche that was still missing after the
reclaim planner and explicit executor landed:

- add a collection lifecycle report
- add explicit reclaim service contracts
- add explicit retention inputs at the service boundary
- persist a default retention policy in storage and use it in service
  operations

## Decisions

These choices were made after checking the existing collection, publish,
snapshot-transfer, and service runtime code.

1. The persisted default retention policy lives on the
   [`CollectionManifest`](../../kayak/collections/collection.mojo), not on
   individual snapshots.
2. Only `default_keep_latest_inactive_count` is persisted for now.
3. Pinned snapshot ids remain request-scoped overrides through
   [`SnapshotRetentionPolicy`](../../kayak/collections/reclaim.mojo).
4. Reclaim execution remains plan-driven at the service layer instead of adding
   implicit deletion.

Why this is sound:

- retention defaults are collection continuity state, not snapshot payload
- request-scoped pins keep the persistence model smaller until there is a real
  hosted requirement for durable pins
- plan-then-execute keeps deletions auditable and preserves stale-plan
  validation

## Implemented

Storage and propagation:

- [`CollectionManifest`](../../kayak/collections/collection.mojo) now carries
  `default_keep_latest_inactive_count`
- [`load_collection_manifest`](../../kayak/collections/collection_store.mojo)
  and [`save_collection_manifest`](../../kayak/collections/collection_store.mojo)
  round-trip that field, defaulting old manifests to `1`
- publication and import/export preserve the collection default in:
  - [`publish_collection_snapshot`](../../kayak/collections/publish.mojo)
  - [`promote_collection_generation`](../../kayak/collections/publish.mojo)
  - [`collection_manifest_for_snapshot_bundle`](../../kayak/collections/snapshot_transfer.mojo)
  - [`merged_collection_manifest_for_import`](../../kayak/collections/snapshot_transfer.mojo)

Service contracts and runtime:

- lifecycle/reclaim contracts live in
  [`kayak/service/lifecycle_contracts.mojo`](../../kayak/service/lifecycle_contracts.mojo)
- runtime entrypoints live in
  [`kayak/service/lifecycle_runtime.mojo`](../../kayak/service/lifecycle_runtime.mojo)
- supported operations:
  - `build_collection_lifecycle_report`
  - `build_reclaim_plan`
  - `execute_reclaim`
  - `update_collection_retention_policy`

JSON and package exports:

- service JSON projections now include lifecycle, reclaim, and retention-policy
  update payloads in [`kayak/service/json.mojo`](../../kayak/service/json.mojo)
- package exports were updated in:
  - [`kayak/service/__init__.mojo`](../../kayak/service/__init__.mojo)
  - [`kayak/__init__.mojo`](../../kayak/__init__.mojo)

## Verified By Tests

Focused suites run after implementation:

- `pixi run test_service_contracts`
- `pixi run test_service_json`
- `pixi run test_service_runtime`
- `pixi run test_collection_storage`
- `pixi run test_collections`
- `pixi run test_snapshot_bundle`

Observed result:
- all focused suites passed

## What This Does Not Claim

This tranche does not prove:

- that collection-level defaults are the final hosted retention model
- that pinned-snapshot overrides should never be persisted
- that background or policy-driven automatic reclaim is ready
- that reclaim policy belongs on snapshots

It only establishes a sounder control plane for the hosted engine with tested,
typed, and JSON-projectable lifecycle behavior.
