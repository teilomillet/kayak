# 2026-04-14: Hosted Engine HTTP Transport

## Claim

Kayak now has a real hosted-engine HTTP transport that wraps the existing
Mojo service runtime instead of reimplementing the engine in Python.

## Why This Change

Before this step, the repo had:
- typed hosted-engine contracts under `kayak/service/`
- JSON projection helpers under `kayak/service/json.mojo`
- no real service process reachable over the network

That meant the service architecture was legible, but not yet deployable.

The goal of this tranche was to close that gap with the thinnest sound
transport:
- keep hosted execution in Mojo
- keep the public local SDK separate
- add one boring HTTP process that proves the service boundary works end to end

## Decision

Implement the transport as:

1. a Mojo-backed Python extension for hosted-engine operations
2. a separate Python package boundary under `python/kayak_engine/`
3. a small stdlib HTTP server on top of that extension

Reason:
- the repo already had a proven pattern for Mojo-backed Python extensions in
  `python/kayak_bridge/_mojo_exact_cpu_bindings.mojo`
- Python's stdlib HTTP stack is good enough for a first transport without
  pulling in a web framework
- placing the server under `kayak_engine` avoids re-blurring the public
  `import kayak` SDK boundary

## Implementation

Added:

- `python/kayak_engine/_mojo_service_bindings.mojo`
- `python/kayak_engine/mojo_service.py`
- `python/kayak_engine/payloads.py`
- `python/kayak_engine/server.py`
- `python/tests/test_hosted_engine_http.py`
- `docs/hosted_engine_http.md`

Updated:

- `pyproject.toml`
- `docs/architecture/service_api.md`
- `docs/readiness_to_5_plan.md`
- `README.md`

## Verified Endpoints

Current HTTP endpoints:

- `GET /health`
- `GET /metrics`
- `POST /v1/collections`
- `POST /v1/documents:upsert`
- `POST /v1/snapshots`
- `POST /v1/search`
- `POST /v1/explain`
- `POST /v1/planned-search`
- `POST /v1/planned-explain`

## Verification

Command run:

```bash
PYTHONPATH=python pixi run python -m unittest python/tests/test_hosted_engine_http.py
```

Observed result:

- passed
- covered:
  - service startup
  - health
  - metrics
  - create collection
  - upsert documents
  - create snapshot
  - exact search
  - exact explain
  - planned search
  - wrong-model rejection

## Important Negative Finding

The initial server used `ThreadingHTTPServer`.

Observed result:
- live search crashed the process even though the binding worked correctly when
  called directly in-process

Interpretation:
- the transport concurrency model was not yet verified safe for the current
  Mojo binding path

Follow-up change:
- switched the first transport to plain `HTTPServer`

Observed result after that change:
- the full networked integration test passed

## Current Limits

Verified limits of this tranche:

- single-process, single-threaded transport
- no auth
- no reclaim/lifecycle HTTP endpoints yet
- no export/import HTTP endpoints yet
- no binary ingest path
- local Mojo CLI still required for the hosted-engine binding build

These are acceptable for this step because the goal was to prove the service
edge, not to finish self-hosting or operator UX.
