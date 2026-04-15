# Hosted Prepared Exact Runtime Service

## Claim

The repo already had a verified local prepared exact-search runtime, but the
hosted HTTP transport still forced callers through stateless search routes that
reloaded snapshot state on every request.

The next justified server step was therefore:

- keep `/v1/search` stateless
- add an explicit hosted prepared exact-runtime surface beside it
- reuse the already-verified process-backed runtime contract
- expose lifecycle and stats explicitly instead of hiding a cache

## Why this was the right scope

The earlier evidence already established three things:

1. the reusable same-snapshot seam is real and valuable
2. the hosted HTTP server was still single-request and stateless
3. the verified safe same-snapshot backend is process-backed, not threaded

That means the sound hosted cut is:

- explicit prepare/reuse by runtime handle
- explicit search and batch routes over that handle
- explicit close and stats routes
- explicit invalidation when reclaim removes referenced snapshots

It does **not** mean:

- make `/v1/search` secretly stateful
- claim that the hosted HTTP transport is now concurrent
- claim that threaded Python-to-Mojo exact search is safe

## Implementation

Added hosted registry module:

- [hosted_prepared_exact_runtime_registry.py](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:1)

This module owns:

- runtime-keyed prepare/reuse
- runtime summaries
- runtime close
- reclaim-driven invalidation

Reason:

- server lifecycle logic should not be smeared across route handlers
- the hosted registry needs one explicit place to define reuse and invalidation

Extended the process runtime:

- [prepared_exact_process_runtime.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_process_runtime.py:329)

Added:

- `search_batch(...)`

Reason:

- the hosted service should reuse the same verified runtime abstraction for
  repeated and batch exact search
- rebuilding batching inside the HTTP handler would create a second runtime
  contract

Extended payload parsing:

- [payloads.py](/Users/teilomillet/Code/kayak/python/kayak_engine/payloads.py:1)

Added:

- exact scoring payload parsing for runtime config
- prepared exact runtime config parsing

Reason:

- the hosted route should accept the same explicit runtime policy knobs already
  verified in the local Python runtime

Extended the hosted HTTP server:

- [server.py](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:1)

Added routes:

- `GET /v1/prepared-exact-runtimes`
- `POST /v1/prepared-exact-runtimes`
- `POST /v1/prepared-exact-runtimes:stats`
- `POST /v1/prepared-exact-runtimes:close`
- `POST /v1/prepared-exact-search`
- `POST /v1/prepared-exact-search-batch`

Important behavior:

- prepare reuses an existing runtime only when
  `(collection_id, tenant_id, namespace_id, snapshot_id, load_text_corpus,
  config)` all match
- `/v1/search` remains stateless and unchanged
- reclaim execution now invalidates hosted runtimes whose pinned `snapshot_id`
  appears in the plan with `retain = false`
- server shutdown closes all active hosted prepared runtimes

## Validation

Commands run:

```bash
python3 -m py_compile \
  python/kayak_engine/hosted_prepared_exact_runtime_registry.py \
  python/kayak_engine/server.py \
  python/kayak_engine/payloads.py \
  python/kayak_engine/prepared_exact_process_runtime.py \
  python/kayak_engine/prepared_exact_types.py

PYTHONPATH=python pixi run pytest python/tests/test_hosted_engine_http.py -q
PYTHONPATH=python pixi run pytest python/tests/test_prepared_exact_search_session.py -q
PYTHONPATH=python pixi run pytest python/tests/test_prepared_exact_search_scheduler.py -q
```

Observed results:

- `python/tests/test_hosted_engine_http.py`: `4 passed`
- `python/tests/test_prepared_exact_search_session.py`: `4 passed`
- `python/tests/test_prepared_exact_search_scheduler.py`: `6 passed`

Warnings that remain:

- the existing Python binding warning about builtin type
  `PreparedExactSearchSession` lacking `__module__`

That warning already existed on the local prepared exact-search surface and did
not affect hosted route correctness in these checks.

## New network checks

Added hosted integration coverage in:

- [test_hosted_engine_http.py](/Users/teilomillet/Code/kayak/python/tests/test_hosted_engine_http.py:1)

What is now verified over the actual HTTP transport:

1. prepare returns one runtime summary with explicit config and zeroed stats
2. preparing the same identity and config reuses the same `runtime_id`
3. hosted prepared exact search matches the stateless exact route
4. hosted prepared exact batch matches repeated stateless exact search
5. hosted runtime stats expose the request and batch counters after real use
6. close removes the handle and future stats/search requests fail with `404`
7. reclaim execution invalidates runtimes pinned to reclaimed snapshots
8. reclaim dry runs do **not** invalidate hosted runtimes

## What this does not prove

- that hosted HTTP is now multi-request concurrent
- that prepared exact runtime policy should be hidden behind `/v1/search`
- that the hosted runtime should support planned search or explain yet
- that a threaded backend is safe
- any new throughput number for the hosted service

The change adds the missing explicit service seam. It does not claim a new
performance figure for the network path.

## Conclusion

What is now true:

- the hosted engine has an explicit exact-search runtime surface over HTTP
- repeated same-snapshot exact reuse is now available without dropping down to
  local Python-only APIs
- runtime policy, stats, and invalidation are explicit and inspectable
- reclaim no longer leaves stale hosted runtime handles behind for removed
  snapshots

What is still true:

- `/v1/search` is still stateless
- the HTTP server is still single-threaded
- if transport-level concurrency becomes the next goal, that should be tackled
  as a separate measured server decision rather than implied by this runtime
  layer
