# Prepared Exact Search Executor

## Claim

For the same-machine, same-snapshot, multi-query serving case, the next module
step was not a transport change. It was to add an explicit worker-local Mojo
execution seam that:

- owns one prepared published snapshot
- owns one exact CPU scoring configuration
- can be reused across many search requests

This gives the repo a real reusable kernel before any Python concurrency API is
chosen.

## Why this module-first seam was justified

Code inspection before editing showed:

- the current prepared-snapshot seam already exists in
  [prepared_snapshot_runtime.mojo](/Users/teilomillet/Code/kayak/kayak/service/prepared_snapshot_runtime.mojo:1)
- the hosted Python edge still creates a fresh `ExactCpuBackend()` per request
  in
  [python/kayak_engine/_mojo_service_bindings.mojo](/Users/teilomillet/Code/kayak/python/kayak_engine/_mojo_service_bindings.mojo:783)
  and
  [python/kayak_engine/_mojo_service_bindings.mojo](/Users/teilomillet/Code/kayak/python/kayak_engine/_mojo_service_bindings.mojo:894)
- exact scorer concurrency controls already live in the canonical Mojo kernel at
  [exact_scoring_config.mojo](/Users/teilomillet/Code/kayak/kayak/scoring/exact_scoring_config.mojo:1)

That meant the sound module-first change was:

- keep `ExactScoringConfig` as the canonical kernel config
- add one small service module that composes:
  - `PreparedSearchSnapshot`
  - `ExactCpuBackend`
- avoid inventing a Python-shaped duplicate config object at the Mojo layer

Reason:

- the module should stay canonical and easy to optimize later
- Python can project into this seam later without becoming the source of truth
- the same executor object is also the right kernel for future process-local
  snapshot caches

## Implementation

Added:

- [prepared_exact_search_executor.mojo](/Users/teilomillet/Code/kayak/kayak/service/prepared_exact_search_executor.mojo:1)

This module owns:

- `PreparedExactSearchExecutor`
- `exact_cpu_backend_for_scoring_config(...)`
- prepare helpers for collection-root and service-root execution with and
  without an explicit `ExactScoringConfig`

The executor intentionally owns the prepared snapshot by move, not by copy.

Reason:

- `PreparedSearchSnapshot` is movable rather than copyable
- worker-local ownership matches the intended serving model more closely anyway

Exported through:

- [kayak/service/__init__.mojo](/Users/teilomillet/Code/kayak/kayak/service/__init__.mojo:1)
- [kayak/__init__.mojo](/Users/teilomillet/Code/kayak/kayak/__init__.mojo:1)

Added focused tests:

- [test_service_prepared_exact_search_executor.mojo](/Users/teilomillet/Code/kayak/tests/test_service_prepared_exact_search_executor.mojo:1)

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_service_prepared_exact_search_executor.mojo
pixi run mojo -I . tests/test_service_prepared_snapshot_runtime.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
```

Observed results:

- `tests/test_service_prepared_exact_search_executor.mojo`: `2/2` passed
- `tests/test_service_prepared_snapshot_runtime.mojo`: `2/2` passed
- `tests/test_service_runtime.mojo`: `26/26` passed

What the focused executor tests verify:

- default executor behavior matches the existing prepared-snapshot runtime path
- explicit serial and explicit parallel scoring configs keep exact-search hits
  identical on the same prepared snapshot

## What is verified and what is not

Verified:

- the repo now has a narrow Mojo kernel for worker-local repeated exact search
- the executor keeps scoring config explicit at the module boundary
- the change is compatible with existing prepared-snapshot and hosted service
  behavior

Not yet verified:

- throughput gains from any particular Python concurrency strategy
- the best worker-count versus intra-request-parallelism tradeoff
- whether a threaded in-process Python surface is safe or worthwhile

So the sound current conclusion is:

- this module seam is the right kernel to optimize and wrap later
- Python transport and concurrency choices should be built on top of it only
  after measuring the same-machine multi-query workload directly
