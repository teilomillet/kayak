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

Verified on `2026-04-15`:

- `GET /v1/prepared-exact-runtimes` works
- `POST /v1/prepared-exact-runtimes` works
- `POST /v1/prepared-exact-runtimes:stats` works
- `POST /v1/prepared-exact-runtimes:close` works
- `POST /v1/prepared-exact-search` works
- `POST /v1/prepared-exact-search-batch` works

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
- hosted same-snapshot reuse now also has an explicit HTTP surface:
  - `POST /v1/prepared-exact-runtimes`
  - `POST /v1/prepared-exact-search`
  - `POST /v1/prepared-exact-search-batch`

Current verified backend:
- the local prepared exact-search runtime only verifies
  `execution_backend="process"`
- in that backend, `concurrency_lane_count` maps to the number of worker
  processes holding prepared same-snapshot state

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
- `GET /v1/prepared-exact-runtimes`
- `POST /v1/prepared-exact-runtimes`
- `POST /v1/prepared-exact-runtimes:stats`
- `POST /v1/prepared-exact-runtimes:close`
- `POST /v1/prepared-exact-search`
- `POST /v1/prepared-exact-search-batch`

Current design choice:
- exact search and explain have a narrower wire shape
- debug-search routes expose the explain payload directly
- planned search, planned debug, and planned explain expose planner-facing knobs
  explicitly
- lifecycle and reclaim routes keep plan-then-execute explicit rather than
  hiding cleanup behind implicit policy
- prepared exact runtime reuse stays explicit through a separate runtime handle
  rather than changing `/v1/search` semantics

Reason:
- ordinary callers should not need to materialize a full `SearchPlan`
- planner-aware callers still need an explicit route into stage-1 selection
- operator routes should stay auditable and predictable at the wire level
- same-snapshot exact reuse is useful, but it should remain visible at the wire
  instead of becoming a hidden per-process cache behind stateless search

## Hosted Prepared Exact Runtime

The hosted transport now exposes one explicit prepared exact-runtime registry.

What it owns:

- process-backed exact runtimes pinned to one
  `(collection_id, tenant_id, namespace_id, snapshot_id)` tuple
- explicit runtime policy:
  - `execution_backend`
  - `concurrency_lane_count`
  - `worker_count`
  - `max_batch_size`
  - `max_batch_wait_ms`
  - exact scoring options
- per-runtime counters and derived averages

What it does not own:

- planned search
- explain
- automatic hidden reuse behind `/v1/search`
- full threaded execution of direct Mojo service routes

Current prepare/reuse rule:

- `POST /v1/prepared-exact-runtimes` prepares a runtime when the
  `(identity, load_text_corpus, config)` key is new
- the same request reuses the existing hosted runtime and returns the same
  `runtime_id`
- preparing one new hosted runtime does not block searches on already-active
  hosted runtimes

Current invalidation rule:

- publishing a newer snapshot does **not** invalidate a runtime pinned to an
  older snapshot
- executing reclaim invalidates hosted runtimes whose pinned `snapshot_id`
  appears in the reclaim plan with `retain = false`
- reclaim dry runs do **not** invalidate hosted runtimes

Reason:

- the pinned-snapshot tests already verified that prepared exact search can stay
  correct across newer publishes
- reclaim is the first operator action that explicitly removes snapshot
  artifacts, so the hosted runtime should not silently survive that policy

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

Prepare one hosted exact runtime:

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
      "max_batch_wait_ms": 25
    }
  }'
```

`load_text_corpus` is optional on this exact-only runtime surface and defaults to
`false`. Set it explicitly only if you need prepared document text loaded for a
future text-dependent extension on the same pinned snapshot.

Prepared exact runtime routes are now the only hosted routes that intentionally
overlap across HTTP requests. They can batch concurrent searches onto one
prepared runtime handle. The generic collection, snapshot, search, and planned
search routes still execute through one dedicated engine thread so the direct
Mojo service bindings remain serialized and predictable.

Search through one hosted prepared runtime:

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

Batch through one hosted prepared runtime:

```bash
curl -sS http://127.0.0.1:8000/v1/prepared-exact-search-batch \
  -H 'content-type: application/json' \
  -d '{
    "runtime_id": "prepared-exact-runtime-0001",
    "requests": [
      {
        "query_model_name": "colbertv2",
        "query": [[1.0, 0.0], [0.0, 1.0]],
        "final_k": 2
      },
      {
        "query_model_name": "colbertv2",
        "query": [[0.0, 1.0], [1.0, 0.0]],
        "final_k": 2
      }
    ]
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

- single-process server
- direct collection, snapshot, stateless exact-search, and planned-search routes
  still execute through one dedicated engine thread
- only prepared exact-runtime routes intentionally overlap across HTTP requests
- extra process lanes still duplicate prepared snapshot memory; concurrency is
  not free
- no auth yet
- no streaming results
- no binary ingest transport
- snapshot transfer is currently limited to local `file://` URIs
- planned search currently supports `best_effort` faithfulness at the HTTP edge

Reason:
- this step solves the deployable-service and operator-surface gap first
- auth, broader concurrency, and richer I/O should stay separate follow-on
  decisions

## Why Concurrency Is Split

The hosted transport now uses `ThreadingHTTPServer`, but it does **not** treat
all routes as equally thread-safe.

Reason:
- direct Mojo service calls from arbitrary handler threads were reproduced as
  unsafe during live validation
- prepared exact-runtime routes were then isolated because they hand work off to
  the verified process-backed runtime
- direct engine routes still go through one dedicated engine thread so the
  service binding stays serialized and predictable

This is the sound epistemic choice:
- allow concurrency only where the runtime contract is already explicit and
  verified
- keep direct Mojo binding calls serialized until that path is proven safe under
  broader threading

Local follow-on note:
- the repo now has a verified process-backed Python runtime for same-snapshot
  exact search
- the earlier scheduler naming remains as a compatibility alias
- it is process-backed on purpose because the background-thread Mojo path
  segfaulted during direct validation
