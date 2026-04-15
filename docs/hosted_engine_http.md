# Hosted Engine HTTP Transport

Status: `implemented non-auth hosted transport`
Date: `2026-04-14`

This note defines the first real HTTP transport for Kayak Engine.

It is intentionally narrow.

Reason:
- the repo already had sound typed service contracts in `kayak/service/`
- the missing piece was a deployable process reachable over the network
- the safest first transport is a thin adapter over the real Mojo engine, not a
  second engine written in Python

## Verified Current Shape

The current transport consists of:

- a Mojo-backed Python extension at
  [python/kayak_engine/_mojo_service_bindings.mojo](../python/kayak_engine/_mojo_service_bindings.mojo)
- a small Python loader at
  [python/kayak_engine/mojo_service.py](../python/kayak_engine/mojo_service.py)
- a stdlib HTTP server at
  [python/kayak_engine/server.py](../python/kayak_engine/server.py)
- a networked integration test at
  [python/tests/test_hosted_engine_http.py](../python/tests/test_hosted_engine_http.py)

Verified on `2026-04-14`:

- the server starts as a standalone process
- `GET /health` works
- `GET /metrics` works
- `POST /v1/collections` works
- `POST /v1/collections:lifecycle` works
- `POST /v1/collections:reclaim-execute` works
- `POST /v1/collections:reclaim-plan` works
- `POST /v1/collections:retention` works
- `POST /v1/debug-search` works
- `POST /v1/documents:delete` works
- `POST /v1/documents:upsert` works
- `POST /v1/snapshots` works
- `POST /v1/snapshots:export` works
- `POST /v1/snapshots:import` works
- `POST /v1/search` works
- `POST /v1/explain` works
- `POST /v1/planned-debug-search` works
- `POST /v1/planned-search` works
- `POST /v1/planned-explain` works

For operator workflows beyond the raw endpoint list, see
[docs/hosted_engine_operator_guide.md](hosted_engine_operator_guide.md).

## Package Boundary

The transport lives under `kayak_engine`, not under the public `kayak` SDK.

Reason:
- `import kayak` is the open local Python SDK for late-interaction programming
- the hosted process is a different product surface with different stability
  and operational concerns
- keeping the package boundary explicit avoids re-blurring SDK and engine

Current consequence:
- `import kayak` remains the SDK
- `python -m kayak_engine.server ...` is the local server startup path
- installed environments also expose `kayak-engine-serve`
- local same-snapshot Python reuse now also has an explicit non-HTTP surface:
  - `prepare_exact_search_session(...)`
  - `prepare_exact_search_runtime(...)`
  - `prepare_exact_search_scheduler(...)` as a compatibility alias

Current verified backend:
- the local prepared exact-search runtime only verifies
  `execution_backend="process"`

Reason:
- the runtime contract can stay stable even if the backend changes later
- the threaded Python-to-Mojo exact-search path was not stable in local testing
- documenting only the verified backend keeps the surface honest

## Startup

Verified local repo path:

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

Current requirement:
- the hosted engine still requires a usable local `mojo` CLI to build the
  service bindings

Reason:
- the current transport compiles a small service extension module at runtime
- the repo has not yet shipped a prebuilt hosted-engine extension artifact

## Endpoints

Current implemented endpoints:

- `GET /health`
- `GET /metrics`
- `POST /v1/collections`
- `POST /v1/collections:lifecycle`
- `POST /v1/collections:reclaim-execute`
- `POST /v1/collections:reclaim-plan`
- `POST /v1/collections:retention`
- `POST /v1/debug-search`
- `POST /v1/documents:delete`
- `POST /v1/documents:upsert`
- `POST /v1/snapshots`
- `POST /v1/snapshots:export`
- `POST /v1/snapshots:import`
- `POST /v1/search`
- `POST /v1/explain`
- `POST /v1/planned-debug-search`
- `POST /v1/planned-search`
- `POST /v1/planned-explain`

Current design choice:
- exact search and explain have a narrower wire shape
- debug-search routes expose the explain payload directly
- planned search, planned debug, and planned explain expose planner-facing knobs
  explicitly
- lifecycle and reclaim routes keep plan-then-execute explicit rather than
  hiding cleanup behind implicit policy

Reason:
- ordinary callers should not need to materialize a full `SearchPlan`
- planner-aware callers still need an explicit route into stage-1 selection
- operator routes should stay auditable and predictable at the wire level

## Example

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

Upsert documents:

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
        "text": "alpha evidence document",
        "metadata": {"topic": "alpha"}
      }
    ]
  }'
```

Create a snapshot:

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

Search exactly:

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

Export a snapshot bundle:

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

## Current Limits

These are verified limits of the current transport:

- single-process, single-threaded server
- no auth yet
- no streaming results
- no binary ingest transport
- snapshot transfer is currently limited to local `file://` URIs
- planned search currently supports `best_effort` faithfulness at the HTTP edge

Reason:
- this step solves the deployable-service and operator-surface gap first
- auth, concurrency, and richer I/O should stay separate follow-on decisions

## Why Single-Threaded For Now

The first implementation intentionally uses `HTTPServer`, not
`ThreadingHTTPServer`.

Reason:
- the Mojo service binding path was validated in-process and over the network
- the threaded stdlib server crashed during live search
- the single-threaded server was then verified by the end-to-end test

This is the sound epistemic choice:
- prefer the narrower transport that is currently verified
- add concurrency only after the binding/runtime path is proven safe under it

Local follow-on note:
- the repo now has a verified process-backed Python runtime for same-snapshot
  exact search
- the earlier scheduler naming remains as a compatibility alias
- it is process-backed on purpose because the background-thread Mojo path
  segfaulted during direct validation
