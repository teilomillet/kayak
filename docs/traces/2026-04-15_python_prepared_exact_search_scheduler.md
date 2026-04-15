# 2026-04-15: Python Prepared Exact Search Scheduler

Historical note:

- this trace records the first process-backed same-snapshot Python batching
  surface
- the canonical naming later moved to the runtime contract documented in
  `2026-04-15_python_prepared_exact_search_runtime_contract.md`

## Claim

Local Python callers can now submit multiple exact searches against one pinned
hosted snapshot through an explicit scheduler surface:

- `prepare_exact_search_scheduler(...)`
- `PreparedExactSearchScheduler.submit(...)`
- `PreparedExactSearchScheduler.search(...)`
- `PreparedExactSearchScheduler.stats()`

The scheduler is implemented with a dedicated local worker **process**, not a
worker thread.

## Why Process, Not Thread

The first implementation hypothesis was a background-thread scheduler over the
prepared-session batch kernel.

That hypothesis was falsified locally.

Observed failure:

- calling the prepared exact-search path from a background Python thread
  segfaulted inside the Mojo extension
- this reproduced both for a single scheduler request and for concurrent
  submitters

The crash stack showed the failure inside the hosted-engine Mojo extension while
executing exact scoring from the scheduler worker thread.

So the threaded scheduler was not a sound foundation for a public Python
surface.

Reason for the replacement design:

- the exact same Python API shape can be preserved
- the actual Mojo call can run on the worker process main thread
- parent-process threads only submit requests and resolve futures
- batching policy stays explicit and measurable

## Implementation

Extended:

- `python/kayak_engine/prepared_search.py`
- `python/kayak_engine/__init__.py`

Added public types:

- `PreparedExactSearchSchedulerConfig`
- `PreparedExactSearchSchedulerStats`
- `PreparedExactSearchScheduler`
- `prepare_exact_search_scheduler(...)`

Scheduler policy remains explicit through:

- `worker_count`
- `max_batch_size`
- `max_batch_wait_ms`
- `scoring`

Execution model:

1. the parent process normalizes exact-search payloads and enqueues them
2. one spawned worker process prepares one pinned snapshot once
3. the worker coalesces near-simultaneous requests into explicit batches
4. the worker executes `PreparedExactSearchSession._search_batch_normalized(...)`
5. the parent listener thread resolves futures and updates stats

## Validation

### New scheduler tests

Command:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_prepared_exact_search_scheduler.py -q
```

Result:

- `3 passed`

What it verifies:

1. scheduler search matches direct prepared-session search
2. concurrent submitters are coalesced into fewer executed batches than raw
   request count
3. closed schedulers reject new submissions

### Existing prepared-session regression test

Command:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_prepared_exact_search_session.py -q
```

Result:

- `4 passed`

### Hosted HTTP regression test

Command:

```bash
PYTHONPATH=python pixi run pytest python/tests/test_hosted_engine_http.py -q
```

Result:

- `2 passed`

### Mojo batch kernel regression test

Command:

```bash
pixi run mojo -I . tests/test_service_prepared_exact_search_batch.mojo
```

Result:

- `2 passed`

## What Is Now True

- local Python callers can pin one snapshot and reuse it directly
- local Python callers can also submit concurrent same-snapshot exact searches
  through one explicit scheduler
- scheduler batching policy and stats are explicit and inspectable

## What Is Still Not True

- the hosted HTTP server is still single-threaded
- the scheduler does not yet expose planner-driven search or explain
- the scheduler result path still emits one Python binding warning about the
  Mojo type `PreparedExactSearchSession` lacking `__module__`

## Guidance

Use:

- `prepare_exact_search_session(...)` when one caller already owns batching
- `prepare_exact_search_scheduler(...)` when many same-process callers need one
  shared same-snapshot execution surface

Do not infer from this trace that arbitrary threaded Python calls into the
hosted-engine Mojo extension are safe. The local evidence in this repo supports
the process-backed scheduler path, not the threaded one.
