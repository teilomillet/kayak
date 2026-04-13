# 2026-04-13: Planner Registry And Benchmark

## Why This Step Exists

The explicit planner layer was already in place, but two pieces were still too
implicit:

- the default ordering logic still lived as code branches instead of a
  declarative registry
- the planner had no dedicated benchmark artifact that recorded both
  `what it chose` and `how that choice measured`

That left two gaps:

- changing planner defaults would be harder to audit than necessary
- benchmark output could show stage-1 quality, but not whether the planner made
  the intended choice for a given snapshot inventory

## What Changed

### Declarative planner registry

Added:

- [kayak/planning/planner_registry.mojo](../../kayak/planning/planner_registry.mojo)
- [kayak/planning/planning_goal.mojo](../../kayak/planning/planning_goal.mojo)

The planner now gets its ordered default candidate-generator preferences from a
registry entry set with explicit per-goal priority and explicit status:

- `promoted`
- `experimental`
- `regression_baseline`
- `benchmark_only`
- `exact_fallback`

Important consequence:

- default planner policy is now inspectable without reading selection control
  flow
- promoted vs benchmark-only generators are visible in both code and JSON
  output

### Mixed-availability guardrail

The snapshot-inventory path is now tested against multi-segment snapshots where
some sidecar families exist on only a subset of segments.

That validates an important hosted-engine invariant:

- planner selection must key off availability on **all** segments for a stage-1
  family, not mere presence somewhere in the snapshot

### Planner benchmark artifact

Added:

- [kayak/benchmarks/planner_benchmark_json.mojo](../../kayak/benchmarks/planner_benchmark_json.mojo)
- [kayak/benchmarks/planner_benchmark_runner.mojo](../../kayak/benchmarks/planner_benchmark_runner.mojo)
- [benchmarks/real_subset_planner_benchmark.mojo](../../benchmarks/real_subset_planner_benchmark.mojo)
- [benchmarks/real_subset_planner_benchmark_smoke.mojo](../../benchmarks/real_subset_planner_benchmark_smoke.mojo)

The new planner benchmark summary records:

- planning goal
- selected candidate-generator kind
- selected candidate-generator status
- planner reason
- available generator kinds
- effective ordered generator list
- measured faithfulness frontier summary for the chosen concrete plan

This is deliberately planner-facing rather than only plan-facing.

Reason:
- we want to measure the explicit hosted-engine decision boundary, not just the
  raw stage-1/search primitive in isolation

## Benchmark Shape

Two entrypoints now exist:

- full public sweep:
  - `pixi run bench_real_subset_planner_benchmark_raw`
- smoke validation:
  - `pixi run bench_real_subset_planner_benchmark_smoke_raw`

The smoke benchmark is intentionally narrower:

- single public slice: `scifact`
- one candidate-window setting
- all four planner goals:
  - `balanced`
  - `latency_first`
  - `native_multivector`
  - `exact_only`

Observed local smoke artifact:

- output path:
  - `.cache/kayak/public_planner_benchmark_smoke.json`
- observed selections:
  - `balanced -> document_proxy`
  - `latency_first -> document_proxy`
  - `native_multivector -> centroid_postings_imputed_flat`
  - `exact_only -> exact_full_scan`

## Validation

Planner and registry tests:

```bash
pixi run mojo -I . tests/test_search_planner.mojo
pixi run mojo -I . tests/test_planner_registry.mojo
pixi run mojo -I . tests/test_snapshot_inventory.mojo
```

Planner benchmark JSON validation:

```bash
pixi run mojo -I . tests/test_planner_benchmark_json.mojo
```

Service/runtime validation:

```bash
pixi run mojo -I . tests/test_service_contracts.mojo
pixi run mojo -I . tests/test_service_json.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
```

Benchmark smoke validation:

```bash
pixi run bench_real_subset_planner_benchmark_smoke_raw
```

Observed result:

- all listed tests passed locally
- the smoke benchmark wrote the expected planner artifact JSON

## Epistemic Boundary

This step proves that `kayak` now has:

- explicit planner registry metadata
- multi-segment availability guardrails
- a planner-native benchmark artifact
- a runnable smoke benchmark for planner selection behavior

This step does **not** prove:

- that the current promoted ordering is final
- that the full public planner sweep is cheap enough for routine local use
- that current native multi-vector promotion should move beyond the current
  `centroid_postings_imputed_flat` default

Those remain empirical questions, but they are now measurable with an explicit
registry seam and a benchmark surface built for that purpose.
