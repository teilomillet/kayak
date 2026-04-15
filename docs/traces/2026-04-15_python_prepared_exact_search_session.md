# 2026-04-15: Python Prepared Exact Search Session

## Claim

The hosted-engine Python layer can now reuse one pinned published snapshot for
repeated exact search from Python without reloading the snapshot on every call,
and it can submit explicit multi-request exact batches over that same prepared
snapshot.

This does **not** make the hosted HTTP server concurrent. It only adds a local
Python API over the already-implemented Mojo prepared-snapshot and batch
execution seams.

## Why This Change

The Mojo module work already added the important kernel:

- `kayak/service/prepared_exact_search_batch.mojo`

But the Python layer still only exposed stateless JSON calls via:

- `python/kayak_engine/_mojo_service_bindings.mojo`
- `python/kayak_engine/server.py`

That meant Python callers could not actually hold a prepared snapshot or submit
same-snapshot multi-query batches without going back through per-call service
loading.

## Implementation

### Shared exact-request normalization

Added:

- `python/kayak_engine/payloads.py`
  - `exact_search_request_payload(...)`

Reason:

- the HTTP server and the new local prepared-search wrapper should normalize the
  same exact-request shape
- this keeps transport semantics aligned and avoids two drifting Python
  contracts for exact search

Refactored:

- `python/kayak_engine/server.py`

The exact `/v1/search`, `/v1/explain`, and `/v1/debug-search` handlers now use
that shared builder instead of hand-building the request dicts inline.

### Mojo binding

Extended:

- `python/kayak_engine/_mojo_service_bindings.mojo`

Added:

- opaque `PreparedExactSearchSession` Python-visible Mojo type
- `prepare_exact_search_session(...)`
- `prepared_exact_search_json(...)`
- `prepared_exact_search_batch_json(...)`

Design choices:

- snapshot preparation stays explicit
- scoring knobs stay explicit per call via Python-provided config payloads
- batch worker count stays explicit
- the binding uses the existing exact request decoder rather than inventing a
  second internal request representation

### Python wrapper

Added:

- `python/kayak_engine/prepared_search.py`

Public surface:

- `ExactScoringOptions`
- `PreparedExactSearchSession`
- `prepare_exact_search_session(...)`

Behavior:

- the session constructor pins one `(service_root, collection_id, tenant_id,
  namespace_id, snapshot_id)` tuple
- `.search(...)` accepts one exact-search payload and reuses the prepared
  snapshot
- `.search_batch(..., worker_count=...)` accepts multiple exact-search payloads
  and forwards them to the Mojo batch kernel
- request identity mismatches are rejected in Python before crossing the bridge

Exported from:

- `python/kayak_engine/__init__.py`

## Validation

### New integration test

Command:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_prepared_exact_search_session.py -q
```

Result:

- `4 passed`

What it verifies:

1. prepared Python exact search matches the existing stateless exact path
2. prepared Python batch search matches repeated prepared single searches
3. a prepared session remains pinned to the original snapshot after a newer
   snapshot is published, while a freshly prepared session sees the new snapshot
4. the prepared Python session rejects request identity drift before crossing
   the Python/Mojo boundary

### Regression check for hosted HTTP transport

Command:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_hosted_engine_http.py -q
```

Result:

- `2 passed`

Reason this matters:

- the HTTP exact-search handlers were refactored to use the new shared request
  payload builder
- this confirms transport behavior stayed intact

### Mojo batch kernel regression check

Command:

```bash
pixi run mojo -I . tests/test_service_prepared_exact_search_batch.mojo
```

Result:

- `2 passed`

Reason this matters:

- the Python wrapper depends on the Mojo prepared-batch kernel remaining sound

## Observed Limitation

The Python test run emits one warning:

- builtin type `PreparedExactSearchSession` has no `__module__` attribute

This did not affect correctness in the exercised paths. It is a Python binding
presentation issue, not a search result issue.

## What Is Now True

- local Python callers can explicitly prepare one snapshot and reuse it
- local Python callers can issue same-snapshot exact batches with explicit
  worker counts and scoring knobs
- snapshot pinning semantics are verified by test

## What Is Still Not True

- the hosted HTTP server is still based on `HTTPServer` and remains
  single-threaded
- there is still no request scheduler or admission controller above the local
  prepared-session batch kernel
- there is still no evidence yet for throughput scaling at the Python wrapper
  layer itself beyond the previously measured Mojo kernel-level batch trace
