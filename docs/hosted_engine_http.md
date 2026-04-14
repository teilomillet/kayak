# Hosted Engine HTTP Transport

Status: `implemented v0 transport`  
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
- `POST /v1/documents:upsert` works
- `POST /v1/snapshots` works
- `POST /v1/search` works
- `POST /v1/explain` works
- `POST /v1/planned-search` works

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
- `POST /v1/documents:upsert`
- `POST /v1/snapshots`
- `POST /v1/search`
- `POST /v1/explain`
- `POST /v1/planned-search`
- `POST /v1/planned-explain`

Current design choice:
- exact search and explain have a narrower wire shape
- planned search and explain expose planner-facing knobs explicitly

Reason:
- ordinary callers should not need to materialize a full `SearchPlan`
- planner-aware callers still need an explicit route into stage-1 selection

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

## Current Limits

These are verified limits of the current transport:

- single-process, single-threaded server
- no auth yet
- no retention, reclaim, import/export, or lifecycle HTTP endpoints yet
- no streaming results
- no binary ingest transport
- planned search currently supports `best_effort` faithfulness at the HTTP edge

Reason:
- this step solves the deployable-service gap first
- the next self-hosting tranche should handle operator concerns separately

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
