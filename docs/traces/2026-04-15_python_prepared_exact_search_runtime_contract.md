# 2026-04-15: Python Prepared Exact Search Runtime Contract

## Claim

The canonical local Python surface for many same-process exact searches against
one pinned hosted snapshot is now a prepared exact-search **runtime**:

- `PreparedExactSearchRuntimeConfig`
- `PreparedExactSearchRuntime`
- `prepare_exact_search_runtime(...)`

The earlier scheduler names remain compatibility aliases:

- `PreparedExactSearchSchedulerConfig`
- `PreparedExactSearchScheduler`
- `prepare_exact_search_scheduler(...)`

Only `execution_backend="process"` is verified today.

## Why This Refactor Was Justified

The first public same-process surface was named around a scheduler mechanism.

That was useful as an implementation description, but it baked one mechanism
into the conceptual contract.

The stronger contract is:

- one pinned snapshot
- explicit batching and scoring policy
- one local runtime for multiple same-process callers
- backend choice kept explicit and replaceable
- explicit concurrency lanes above the batch kernel

Reason:

- runtime semantics are more stable than any one implementation mechanism
- backend-neutral naming lets the code evolve without pretending that threads,
  processes, or future runtimes are all equivalent today
- compatibility aliases preserve working user code while the design becomes
  more honest and easier to extend

## Evidence Behind Backend Support

The threaded Python-to-Mojo hypothesis was already falsified locally during the
previous scheduler slice.

Observed evidence:

- calling the prepared exact-search path from a background Python thread
  segfaulted inside the hosted-engine Mojo extension
- this reproduced even for a single background worker, not only for
  multi-request load

So the sound runtime contract is:

- support `execution_backend="process"`
- reject unverified backends early
- keep compatibility aliases, but do not imply thread safety
- expose independent runtime-lane count explicitly instead of implying that the
  batch-kernel `worker_count` already means multiple runtime workers

## Implementation

Refactored the Python package into narrower modules:

- `python/kayak_engine/prepared_exact_types.py`
  - shared scoring/runtime config and validation helpers
- `python/kayak_engine/prepared_exact_session.py`
  - one pinned snapshot session and request-identity normalization
- `python/kayak_engine/prepared_exact_process_runtime.py`
  - the current verified process-backed runtime implementation
- `python/kayak_engine/prepared_exact_runtime.py`
  - canonical runtime surface plus compatibility aliases
- `python/kayak_engine/prepared_search.py`
  - small facade for the public package import path

This split was justified because the prior `prepared_search.py` had grown into
an 818-line mixed-responsibility module.

## Validation

### Runtime Python tests

Command:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_prepared_exact_search_scheduler.py -q
```

Result:

- `5 passed`

What it verifies:

1. scheduler names are aliases to the runtime surface
2. unsupported backends such as `"thread"` are rejected
3. non-positive `concurrency_lane_count` is rejected
4. runtime search matches direct prepared-session search
5. a multi-lane runtime configuration executes correctly
6. concurrent submitters are coalesced into fewer executed batches than raw
   request count
7. closed runtimes reject new submissions

### Prepared-session regression test

Command:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_prepared_exact_search_session.py -q
```

Result:

- `4 passed`

## What Is Now True

- the local Python API is explicit about runtime semantics rather than centering
  implementation mechanism in the naming
- only the process backend is represented as supported today
- the runtime now exposes `concurrency_lane_count` explicitly so same-snapshot
  concurrent search is not limited to one process lane by construction
- compatibility aliases preserve the earlier scheduler surface
- the code is easier to inspect because session, config, runtime contract, and
  process backend are now separated

## What Is Not Yet True

- the hosted HTTP server is still not being claimed as concurrent here
- the runtime does not prove that arbitrary threaded calls into the Mojo
  extension are safe
- multiple runtime backends are not implemented or benchmarked yet

## Guidance

Use:

- `prepare_exact_search_session(...)` when one caller already owns batching
- `prepare_exact_search_runtime(...)` when many same-process callers share one
  pinned snapshot
- the scheduler names only when you need compatibility with earlier code or
  docs

Do not infer from this trace that backend choice is arbitrary. The local
evidence currently supports the process backend only.
