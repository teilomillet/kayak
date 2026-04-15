# Hosted Prepared Runtime Threaded Transport

## Claim

After narrowing the prepared exact-search load contract, the next same-machine
question was:

- can the hosted HTTP transport actually overlap multiple prepared exact-search
  requests on the same pinned snapshot
- can that overlap feed the existing process-backed runtime batcher instead of
  forcing callers through one single-threaded request loop
- can we do that without making the direct Mojo service routes run on arbitrary
  handler threads

The justified target was:

- keep generic hosted engine routes serialized through one explicit engine
  execution lane
- allow only the hosted prepared exact-runtime routes to overlap across HTTP
  requests
- make the hosted runtime registry safe for concurrent search vs close/summary
  traffic

## Why this scope was justified

The earlier hosted transport was still single-request:

- [KayakEngineHttpServer](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:101)
  previously inherited plain `HTTPServer`

But simply switching to `ThreadingHTTPServer` was not enough. During
implementation, direct stateless `/v1/search` calls after prepared-runtime
creation reproduced a hard server crash when the Mojo service binding was
invoked from arbitrary handler threads.

That means the safe transport rule is not "make everything threaded." It is:

- prepared exact-runtime routes may overlap because they hand work off to the
  verified process-backed runtime
- direct Mojo service calls must still execute through one dedicated engine
  thread

## Implementation

Added a dedicated hosted engine executor:

- [HostedEngineExecutor](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:46)

This owns:

- loading the Mojo-backed service module once on one dedicated thread
- executing direct engine callbacks on that same thread

Updated the hosted server:

- [KayakEngineHttpServer](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:101)
  now inherits `ThreadingHTTPServer`
- non-prepared routes are funneled through
  [engine_executor.call(...)](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:137)
  and
  [server.py:189](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:189)
- prepared exact-runtime routes are still allowed to execute directly in handler
  threads because they only touch the hosted runtime registry and the
  process-backed runtime surface

Strengthened hosted runtime registry concurrency:

- [HostedPreparedExactRuntimeRegistry](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:60)
  now uses a condition-protected registry
- concurrent search and search-batch acquire/release explicit operation leases at
  [hosted_prepared_exact_runtime_registry.py:161](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:161)
  and
  [hosted_prepared_exact_runtime_registry.py:171](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:171)
- close waits for in-flight operations to drain before runtime shutdown at
  [hosted_prepared_exact_runtime_registry.py:124](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:124)

This keeps:

- prepared exact runtime overlap explicit and local
- direct engine semantics serialized and predictable

## Validation

Correctness checks:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_hosted_engine_prepared_exact_runtime_http.py -q
PYTHONPATH=python pixi run pytest \
  python/tests/test_prepared_exact_search_session.py \
  python/tests/test_prepared_exact_search_scheduler.py -q
```

Observed:

- `python/tests/test_hosted_engine_prepared_exact_runtime_http.py`: `3 passed`
- `python/tests/test_prepared_exact_search_session.py` plus
  `python/tests/test_prepared_exact_search_scheduler.py`: `11 passed`

The new hosted transport regression is:

- [test_network_prepared_exact_runtime_coalesces_concurrent_search_requests(...)](/Users/teilomillet/Code/kayak/python/tests/test_hosted_engine_prepared_exact_runtime_http.py:238)

What it verifies:

- multiple concurrent HTTP `POST /v1/prepared-exact-search` requests can overlap
- responses still match the stateless exact route
- hosted runtime stats show fewer executed batches than submitted requests
- hosted runtime stats show `max_observed_batch_size > 1`

## Measurement

Command run:

```bash
PYTHONPATH=python pixi run python - <<'PY'
# script staged the BrowseComp+ gold-slice task into one hosted server,
# prepared one hosted exact runtime, then measured 32 sequential prepared
# exact-search HTTP requests versus 32 concurrent prepared exact-search HTTP
# requests against that same runtime across 3 repeats.
PY
```

Saved output:

- [.cache/kayak/hosted_prepared_runtime_concurrency_browsecomp_gold.txt](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_concurrency_browsecomp_gold.txt:1)

Measured setup:

- dataset: `Tevatron/browsecomp-plus/gold-slice`
- request pool: `32`
- hosted prepared runtime config:
  - `concurrency_lane_count=1`
  - `worker_count=2`
  - `max_batch_size=32`
  - `max_batch_wait_ms=25`

## Result

From
[hosted_prepared_runtime_concurrency_browsecomp_gold.txt](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_concurrency_browsecomp_gold.txt:1):

- serial wall-clock repeats:
  - `1.2123947500949726 s`
  - `1.2009800829691812 s`
  - `1.2160562919452786 s`
- concurrent wall-clock repeats:
  - `0.31830412498675287 s`
  - `0.29298145801294595 s`
  - `0.2566320840269327 s`

Median comparison:

- sequential hosted prepared search: `1.212395 s`
- concurrent hosted prepared search: `0.292981 s`
- derived speedup: `4.14x`

Final hosted runtime stats from the same run:

- `submitted_request_count`: `208`
- `executed_batch_count`: `123`
- `max_observed_batch_size`: `27`
- `max_observed_queue_depth`: `29`

Interpretation:

- the hosted transport is no longer the limiting one-request loop for prepared
  exact runtime search
- concurrent callers can now feed one prepared runtime handle and actually
  exercise the existing runtime batcher
- same-snapshot overlap at the HTTP boundary is now materially better without
  multiplying snapshot loads through separate hosted processes

## Important Uncertainty

This is one local-host measurement, not a quiet-wrapper benchmark.

So it is useful for proving the direction and for debunking "the hosted
transport is still fully serialized," but it is not strong enough to claim a
universal throughput multiplier across machines or datasets.

## What this does not prove

- that all hosted routes are now concurrent
- that direct stateless `/v1/search` or planned-search routes should be made
  fully threaded
- that the same speedup holds on larger collections, different hardware, or
  remote storage

## Conclusion

What is now true:

- hosted prepared exact-runtime routes can overlap across HTTP requests
- those overlapped requests can batch on one pinned runtime
- direct Mojo engine routes remain serialized through one dedicated engine
  thread

That is the right shape for the current repo:

- safe concurrency where the runtime contract is already explicit
- no hidden policy
- no claim that the entire hosted service is magically thread-safe
