# 2026-04-14: hosted engine HTTP non-auth ops parity

## Claim

The hosted HTTP surface now exposes the remaining existing non-auth service
operations instead of stopping at create, upsert, snapshot, and search.

## Why This Was Necessary

Before this tranche, the repository already had typed Mojo contracts and runtime
implementations for:
- delete documents
- retention updates
- lifecycle reporting
- reclaim plan and execute
- snapshot export and import
- debug-search variants

But the network transport still omitted them, which meant the operator story was
weaker than the engine contracts themselves.

## What Changed

Transport surface added:
- `POST /v1/documents:delete`
- `POST /v1/collections:retention`
- `POST /v1/collections:lifecycle`
- `POST /v1/collections:reclaim-plan`
- `POST /v1/collections:reclaim-execute`
- `POST /v1/snapshots:export`
- `POST /v1/snapshots:import`
- `POST /v1/debug-search`
- `POST /v1/planned-debug-search`

Service-contract cleanup:
- delete now has an explicit `DeleteDocumentsResponse` JSON projection
- snapshot import now validates bundle identity against the request instead of
  ignoring `snapshot_id`

Documentation added:
- [../hosted_engine_http.md](../hosted_engine_http.md)
- [../hosted_engine_operator_guide.md](../hosted_engine_operator_guide.md)

## Evidence

Verified locally:

- `pixi run mojo -I . tests/test_service_json.mojo`
- `pixi run mojo -I . tests/test_service_runtime.mojo`
- `PYTHONPATH=python pixi run python -m unittest python/tests/test_hosted_engine_http.py`

The HTTP integration test now covers:
- debug-search routes
- delete
- retention update
- lifecycle reporting
- reclaim plan
- reclaim dry-run and apply
- snapshot export
- snapshot import into a second service root

## Remaining Limits

- no auth yet
- local `file://` snapshot transfer only
- single-process, single-threaded server
- no binary ingest transport
- no prebuilt hosted-engine extension artifact
