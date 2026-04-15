# Hosted Engine Operator Guide

Status: `non-auth local operator guide`  
Date: `2026-04-14`

This note explains how to run, observe, snapshot, reclaim, export, and restore
the hosted engine over the current HTTP surface.

It is intentionally local-first.

Reason:
- the current hosted transport is verified in one process on one machine
- snapshot transfer is currently file-based
- auth is still explicitly out of scope

## Preconditions

Verified requirements:
- Python 3.11 in the active environment
- a usable `mojo` CLI reachable either from the active environment, from
  `PATH`, or through `pixi run mojo`
- a writable service root

Verified startup path from the repo:

```bash
PYTHONPATH=python pixi run python -m kayak_engine.server \
  --root ./.state/kayak-engine \
  --host 127.0.0.1 \
  --port 8000
```

Installed-package path:

```bash
kayak-engine-serve --root ./.state/kayak-engine --port 8000
```

## Health And Metrics

Probe liveness:

```bash
curl -sS http://127.0.0.1:8000/health
```

Inspect coarse operational counters:

```bash
curl -sS http://127.0.0.1:8000/metrics
```

Current verified metrics include collection, snapshot, segment, document, and
vector counts.

## Basic Collection Flow

Create a collection:

```bash
curl -sS http://127.0.0.1:8000/v1/collections \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "model_name": "colbertv2",
    "vector_dim": 2
  }'
```

Upsert encoded documents:

```bash
curl -sS http://127.0.0.1:8000/v1/documents:upsert \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "documents": [
      {
        "doc_id": "doc-a",
        "vectors": [[1.0, 0.0], [0.0, 1.0]],
        "text": "alpha evidence document"
      }
    ]
  }'
```

Seal a snapshot:

```bash
curl -sS http://127.0.0.1:8000/v1/snapshots \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "snapshot_id": "snapshot-0001",
    "reason": "initial publish"
  }'
```

Delete draft documents:

```bash
curl -sS http://127.0.0.1:8000/v1/documents:delete \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "doc_ids": ["doc-a"]
  }'
```

Semantics to keep in mind:
- snapshots are immutable search targets
- delete operates on the current draft state, not on already-published snapshots
- search still requires an explicit `snapshot_id`

## Search And Debug

Exact search:

```bash
curl -sS http://127.0.0.1:8000/v1/search \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "snapshot_id": "snapshot-0001",
    "query_model_name": "colbertv2",
    "query": [[1.0, 0.0], [0.0, 1.0]],
    "final_k": 2
  }'
```

Exact debug search:

```bash
curl -sS http://127.0.0.1:8000/v1/debug-search \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "snapshot_id": "snapshot-0001",
    "query_model_name": "colbertv2",
    "query_text": "alpha evidence",
    "query": [[1.0, 0.0], [0.0, 1.0]],
    "final_k": 2
  }'
```

Planner-driven debug search:

```bash
curl -sS http://127.0.0.1:8000/v1/planned-debug-search \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "snapshot_id": "snapshot-0001",
    "query_model_name": "colbertv2",
    "query_text": "alpha evidence",
    "query": [[1.0, 0.0], [0.0, 1.0]],
    "final_k": 2,
    "candidate_k": 4,
    "goal": "balanced",
    "preferred_candidate_generator_kinds": ["document_proxy", "exact_full_scan"]
  }'
```

## Prepared Exact Runtime Reuse

Prepare one explicit same-snapshot exact runtime:

```bash
curl -sS http://127.0.0.1:8000/v1/prepared-exact-runtimes \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "snapshot_id": "snapshot-0001",
    "config": {
      "execution_backend": "process",
      "concurrency_lane_count": 1,
      "worker_count": 2,
      "max_batch_size": 8,
      "max_batch_wait_ms": 25,
      "max_outstanding_request_count": 128
    }
  }'
```

Search through that prepared runtime:

```bash
curl -sS http://127.0.0.1:8000/v1/prepared-exact-search \
  -H 'content-type: application/json' \
  -d '{
    "runtime_id": "prepared-exact-runtime-0001",
    "request": {
      "query_model_name": "colbertv2",
      "query": [[1.0, 0.0], [0.0, 1.0]],
      "final_k": 2
    }
  }'
```

Inspect runtime counters:

```bash
curl -sS http://127.0.0.1:8000/v1/prepared-exact-runtimes:stats \
  -H 'content-type: application/json' \
  -d '{
    "runtime_id": "prepared-exact-runtime-0001"
  }'
```

Operational rules:

- `max_outstanding_request_count` bounds accepted-but-not-yet-finished requests
  on one runtime
- if a new request would exceed that limit, the hosted surface returns HTTP
  `429` immediately instead of queueing forever
- `rejected_request_count` and `current_pending_request_count` in runtime stats
  show whether the runtime is currently dropping work or draining normally
- omitting `max_outstanding_request_count` is allowed; the runtime derives a
  default from lane count and batch size and reports that resolved value back in
  the runtime summary

## Retention, Lifecycle, And Reclaim

Update the collection default retention policy:

```bash
curl -sS http://127.0.0.1:8000/v1/collections:retention \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "default_keep_latest_inactive_count": 0
  }'
```

Read lifecycle state with an ephemeral override:

```bash
curl -sS http://127.0.0.1:8000/v1/collections:lifecycle \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "policy_override": {
      "keep_latest_inactive_count": 0,
      "pinned_snapshot_ids": ["snapshot-0001"]
    }
  }'
```

Build a reclaim plan:

```bash
curl -sS http://127.0.0.1:8000/v1/collections:reclaim-plan \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search"
  }'
```

Execute that reclaim plan as a dry run first:

```bash
curl -sS http://127.0.0.1:8000/v1/collections:reclaim-execute \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "dry_run": true,
    "plan": { "...": "copy the exact plan payload from reclaim-plan" }
  }'
```

Important safety rule:
- reclaim is intentionally plan-then-execute
- the HTTP edge does not hide cleanup behind a policy-only trigger
- if the plan is stale or mismatched, execution fails rather than silently
  reclaiming a different target

## Snapshot Export And Restore

Export a snapshot bundle to a local directory:

```bash
curl -sS http://127.0.0.1:8000/v1/snapshots:export \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "snapshot_id": "snapshot-0001",
    "bundle_uri": "file:///tmp/kayak-bundle"
  }'
```

Restore that bundle into another service root:

```bash
curl -sS http://127.0.0.1:8000/v1/snapshots:import \
  -H 'content-type: application/json' \
  -d '{
    "collection_id": "news",
    "tenant_id": "tenant-a",
    "namespace_id": "search",
    "snapshot_id": "snapshot-0001",
    "source_uri": "file:///tmp/kayak-bundle"
  }'
```

Verified current limits:
- both export and import are local `file://` flows
- import validates bundle `collection_id`, `tenant_id`, `namespace_id`, and
  `snapshot_id` against the request before writing
- restore is identity-preserving; it is not a cross-collection rename tool

## Failure Notes

These notes are derived from the current code path, not from a fault-injection
matrix.

Verified execution order in
[kayak/service/runtime.mojo](../kayak/service/runtime.mojo):
- snapshot creation seals a segment, publishes the snapshot, then compacts the
  draft state
- reclaim execution is explicit and separate from lifecycle reporting

Operational implication:
- if a process dies before snapshot publish, the old published snapshot remains
  the visible search target
- if a process dies after publish but before draft compaction, the published
  snapshot may already be visible while draft cleanup remains incomplete
- after an interrupted reclaim attempt, inspect `/health`, `/metrics`, and the
  lifecycle report before retrying with a fresh plan

What is still not verified:
- a formal crash-consistency matrix for seal, publish, and reclaim
- background repair or automatic cleanup

## Upgrade And Rollback Note

The current safest operator pattern is:

1. export the snapshot you care about with `snapshots:export`
2. bring up the new service root or upgraded environment
3. import with `snapshots:import`
4. verify with `/health`, `/metrics`, and one explicit `/v1/search`

Reason:
- snapshot bundles are the current explicit portability unit
- rollback can reuse the same exported bundle without inventing hidden migration
  state

## Current Limits

- no auth yet
- single-process, single-threaded server
- no streaming results
- no binary ingest transport
- no prebuilt hosted-engine extension artifact yet
