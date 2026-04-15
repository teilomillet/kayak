# Prepared Exact Runtime Backpressure

## Claim

After adding hosted overlap and prepare-lock-scope fixes, one operational gap
still remained for same-snapshot multi-user serving:

- prepared exact runtimes accepted unbounded in-memory work
- pending work had no explicit admission cap
- overload was not distinguished from execution failure

The justified target was:

- add one explicit runtime admission limit
- reject excess work immediately instead of silently queueing forever
- surface overload separately in runtime stats and hosted HTTP behavior

## Why this scope was justified

The current process runtime already tracks pending work internally at
[prepared_exact_process_runtime.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_process_runtime.py:253),
but before this change that count was only observational.

That was acceptable for local experimentation, but not for the production shape
we have now started to verify:

- multiple concurrent hosted callers can feed one runtime handle
- runtime prepare no longer blocks searches on active runtimes

Once those two are true, unbounded admission becomes the next obvious failure
mode. The sound rule is not "add more lanes and hope." It is:

- bound accepted work explicitly
- reject above that bound quickly
- make the rejection visible to operators

## Implementation

Added explicit admission policy to
[PreparedExactSearchRuntimeConfig](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_types.py:45):

- new field: `max_outstanding_request_count`
- `0` means "derive a default" rather than "unbounded"

Current derived default in
[prepared_exact_types.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_types.py:204)
is:

- `max(128, concurrency_lane_count * max_batch_size * 4)`

Reason:

- this keeps the default bounded
- it scales with the amount of work one runtime can actually consume
- it still leaves room for short bursts and batching

Added explicit overload error:

- [PreparedExactSearchRuntimeOverloadedError](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_types.py:10)

Changed runtime admission in
[prepared_exact_process_runtime.py](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_process_runtime.py:390):

- single-request and batch-request admission now use one shared helper
- batch admission is atomic: the whole batch is admitted or the whole batch is
  rejected
- overload increments `rejected_request_count` without polluting
  `failed_request_count`

Added runtime stats:

- `rejected_request_count`
- `current_pending_request_count`

See
[prepared_exact_runtime_stats_json(...)](/Users/teilomillet/Code/kayak/python/kayak_engine/prepared_exact_types.py:183).

Hosted transport now maps runtime overload to HTTP `429` in
[server.py](/Users/teilomillet/Code/kayak/python/kayak_engine/server.py:247).

## Validation

Commands run:

```bash
python3 -m py_compile \
  python/kayak_engine/prepared_exact_types.py \
  python/kayak_engine/payloads.py \
  python/kayak_engine/prepared_exact_process_runtime.py \
  python/kayak_engine/prepared_exact_runtime.py \
  python/kayak_engine/prepared_search.py \
  python/kayak_engine/__init__.py \
  python/kayak_engine/server.py \
  python/kayak_engine/hosted_prepared_exact_runtime_registry.py \
  python/tests/test_prepared_exact_search_scheduler.py \
  python/tests/test_hosted_engine_prepared_exact_runtime_http.py
PYTHONPATH=python pixi run pytest \
  python/tests/test_prepared_exact_search_scheduler.py -q
PYTHONPATH=python pixi run pytest \
  python/tests/test_hosted_engine_prepared_exact_runtime_http.py -q
```

Observed:

- `python/tests/test_prepared_exact_search_scheduler.py`: `10 passed, 1 warning`
- `python/tests/test_hosted_engine_prepared_exact_runtime_http.py`: `4 passed`

New local regressions:

- [test_runtime_rejects_overload_above_max_outstanding_request_count(...)](/Users/teilomillet/Code/kayak/python/tests/test_prepared_exact_search_scheduler.py:294)
- [test_runtime_rejects_batch_atomically_when_batch_exceeds_admission_limit(...)](/Users/teilomillet/Code/kayak/python/tests/test_prepared_exact_search_scheduler.py:330)

New hosted regression:

- [test_network_prepared_exact_runtime_returns_429_under_overload(...)](/Users/teilomillet/Code/kayak/python/tests/test_hosted_engine_prepared_exact_runtime_http.py:389)

## Measurement

Ran one live hosted overload measurement and saved the result to:

- [.cache/kayak/hosted_prepared_runtime_overload_contract.txt](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_overload_contract.txt:1)

Setup:

- one hosted prepared runtime
- `concurrency_lane_count=1`
- `worker_count=1`
- `max_batch_size=8`
- `max_batch_wait_ms=250`
- `max_outstanding_request_count=1`
- two concurrent `POST /v1/prepared-exact-search` requests per repeat

Result:

- accepted latency median: `0.252344 s`
- rejected latency median: `0.000930 s`
- final runtime stats:
  - `submitted_request_count`: `5`
  - `rejected_request_count`: `5`
  - `completed_request_count`: `5`
  - `failed_request_count`: `0`

Interpretation:

- accepted work still follows the configured runtime batching path
- overload is rejected quickly instead of becoming hidden queue growth
- operator-visible stats now separate overload from execution failure

## Important Uncertainty

This trace verifies the admission contract and the hosted `429` behavior.

It does not prove that the current default
`max(128, concurrency_lane_count * max_batch_size * 4)` is the best universal
operating point. That value is a safe starting prior, not a final law.

The next step, if needed, is to sweep `max_outstanding_request_count` against
real burst patterns and memory budgets on larger collections.

## Conclusion

Prepared exact runtimes now have a real operating envelope:

- admission is bounded
- overload is explicit
- hosted callers see `429` instead of silent queue growth
- stats distinguish dropped work from failed execution

That is the right next step for same-snapshot multi-user serving because it
makes concurrency measurable and safe before we attempt deeper memory-topology
changes.
