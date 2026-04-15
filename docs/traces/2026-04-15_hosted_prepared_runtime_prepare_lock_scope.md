# Hosted Prepared Runtime Prepare Lock Scope

## Claim

After enabling threaded hosted prepared-runtime transport, one concurrency risk
was still left in the registry layer:

- could preparing one new hosted exact runtime stall searches and summaries on
  already-active runtimes because the registry lock stayed held across full
  runtime construction

The justified target was:

- keep runtime reuse explicit by key
- keep close semantics explicit
- remove registry head-of-line blocking during expensive `prepare_runtime(...)`
  work

## Why this scope was justified

The hosted prepared-runtime surface is supposed to let one pinned snapshot keep
serving while other same-server activity happens around it.

But runtime preparation is not cheap enough to treat as a trivial critical
section. Even on the current BrowseComp gold slice, the real hosted prepare
measurement below still spends about `69 ms` per new runtime:

- [.cache/kayak/hosted_prepared_runtime_prepare_hol_browsecomp_gold.txt](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_prepare_hol_browsecomp_gold.txt:1)

Holding the registry condition across that work would not improve correctness.
It would only serialize unrelated runtime traffic behind prepare-time startup.

## Implementation

The registry now splits identity/lifecycle protection from expensive runtime
construction:

- explicit prepare-in-flight tracking via `_preparing_keys` and `_closed` at
  [hosted_prepared_exact_runtime_registry.py](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:71)
- `prepare_runtime(...)` now:
  - reserves the key under the condition
  - constructs and waits for the runtime outside the condition
  - then publishes the runtime back under the condition
  at
  [hosted_prepared_exact_runtime_registry.py](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:99)
- if registry shutdown wins the race, the freshly prepared runtime is closed
  instead of being published late:
  [hosted_prepared_exact_runtime_registry.py](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:160)
- `close_all()` now marks the registry closed before draining active entries so
  in-flight prepares cannot repopulate it:
  [hosted_prepared_exact_runtime_registry.py](/Users/teilomillet/Code/kayak/python/kayak_engine/hosted_prepared_exact_runtime_registry.py:217)

This keeps:

- same-key prepare reuse explicit
- active-runtime search/summary paths independent from unrelated prepare work
- shutdown behavior explicit instead of relying on timing

## Validation

Added focused registry regressions with a fake runtime:

- [test_prepare_runtime_does_not_block_existing_runtime_search(...)](/Users/teilomillet/Code/kayak/python/tests/test_hosted_prepared_exact_runtime_registry.py:62)
- [test_close_all_finishes_while_runtime_prepare_is_in_flight(...)](/Users/teilomillet/Code/kayak/python/tests/test_hosted_prepared_exact_runtime_registry.py:159)

Commands run:

```bash
python3 -m py_compile \
  python/kayak_engine/hosted_prepared_exact_runtime_registry.py \
  python/tests/test_hosted_prepared_exact_runtime_registry.py
PYTHONPATH=python pixi run pytest \
  python/tests/test_hosted_prepared_exact_runtime_registry.py -q
PYTHONPATH=python pixi run pytest \
  python/tests/test_hosted_engine_prepared_exact_runtime_http.py \
  python/tests/test_prepared_exact_search_session.py \
  python/tests/test_prepared_exact_search_scheduler.py -q
```

Observed:

- `python/tests/test_hosted_prepared_exact_runtime_registry.py`: `2 passed`
- hosted/runtime/session regressions: `14 passed, 1 warning`

## Measurement

Ran a real hosted overlap measurement on the BrowseComp gold slice and saved the
result to:

- [.cache/kayak/hosted_prepared_runtime_prepare_hol_browsecomp_gold.txt](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_prepare_hol_browsecomp_gold.txt:1)

What the script did:

- start one hosted engine server
- load the BrowseComp gold slice into one collection
- prepare one baseline runtime on `snapshot-0001`
- measure baseline latency for one prepared exact-search request
- then repeatedly:
  - publish a new snapshot id
  - start preparing a second runtime for that new snapshot in a background
    thread
  - issue a search against the already-active baseline runtime while that
    prepare is still in flight

## Result

From
[hosted_prepared_runtime_prepare_hol_browsecomp_gold.txt](/Users/teilomillet/Code/kayak/.cache/kayak/hosted_prepared_runtime_prepare_hol_browsecomp_gold.txt:1):

- baseline prepared-search median: `0.030406 s`
- overlap prepared-search median while another runtime prepared: `0.030884 s`
- overlap prepare median: `0.068972 s`
- derived overlap slowdown: `1.016x`

Interpretation:

- active prepared-runtime search latency stayed essentially flat while another
  runtime was being prepared
- the new runtime prepare itself still costs real time
- that prepare cost is no longer showing up as a search stall on the active
  runtime path

## Important Uncertainty

This measurement does not include a before/after wall-clock comparison from the
old registry implementation.

So the latency improvement itself is established by:

- direct code inspection of the old lock scope
- the new fake-runtime regression that would have blocked under the old design
- the real hosted overlap measurement above showing active-runtime searches now
  remain near baseline while a second prepare is in flight

The exact benefit size on larger collections will scale with real prepare cost.

## Conclusion

The hosted prepared-runtime transport is now stronger in one more important
same-machine concurrency case:

- concurrent search on an active runtime no longer waits behind unrelated hosted
  runtime preparation
- same-key reuse and close semantics remain explicit
- shutdown no longer risks late runtime publication after close has begun
