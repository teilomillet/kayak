# 2026-04-13: Stage-2-Swept Planner Benchmark

## Why This Step Exists

The repository already had:

- a planner benchmark artifact
- a planner-evidence artifact
- a mirrored text-sidecar path for planner evidence

But the two benchmark surfaces still differed in one important way:

- planner evidence could sweep multiple stage-2 operators
- planner benchmark only measured the planner's selected stage-1 under its
  default stage-2

That made comparison weaker than necessary. We could compare planner selection
and planner evidence on the same candidate-generator axis, but not yet on the
same stage-2 axis.

## What Changed

Updated:

- [kayak/benchmarks/planner_benchmark_json.mojo](../../kayak/benchmarks/planner_benchmark_json.mojo)
- [kayak/benchmarks/planner_benchmark_runner.mojo](../../kayak/benchmarks/planner_benchmark_runner.mojo)
- [benchmarks/browsecomp_plus_gold_planner_benchmark.mojo](../../benchmarks/browsecomp_plus_gold_planner_benchmark.mojo)

### Planner benchmark summaries are now stage-2-explicit

`PlannerBenchmarkSummary` now records:

- `planning_goal`
- `stage2_kind`
- selected candidate-generator metadata
- measured frontier summary

`build_planner_benchmark_summary(...)` now accepts an explicit
`Stage2Operator`, applies it to the selected plan, and measures that concrete
plan.

This mirrors the benchmark/evidence separation already used in planner
evidence:

- planner still selects the stage-1 generator
- the benchmark can now intentionally hold stage-2 fixed or sweep it

### The runner has an optional stage-2 sweep

`PlannerBenchmarkRunOptions` now includes:

- `include_clause_text_stage2_when_text_available`

Behavior:

- default and smoke runs remain cheap and only measure
  `exact_late_interaction`
- datasets with mirrored text sidecars can opt into an additional
  `clause_text` sweep

The current runner only loads document texts when that option is enabled, so
the default path does not pay hidden text-loading overhead.

### BrowseComp gold benchmark enables the extra axis

The gold planner benchmark entrypoint now opts into the clause-text sweep.

Because the mirror path already supports text sidecars, the output artifact now
contains summaries for:

- `exact_late_interaction`
- `clause_text`

## Validation

Planner benchmark unit and integration tests:

```bash
pixi run mojo -I . tests/test_planner_benchmark_json.mojo
```

Default runner smoke path:

```bash
pixi run bench_real_subset_planner_benchmark_smoke_raw
```

Gold stage-2-swept benchmark:

```bash
pixi run bench_browsecomp_plus_gold_planner_benchmark_raw
```

Observed result:

- all listed commands passed locally
- `.cache/kayak/public_planner_benchmark_smoke.json` contains:
  - `4` summaries
  - only `stage2_kind = exact_late_interaction`
- `.cache/kayak/browsecomp_plus_gold_planner_benchmark.json` contains:
  - `40` summaries
  - `stage2_kind` values:
    - `exact_late_interaction`
    - `clause_text`

## Epistemic Boundary

This step proves:

- planner benchmark and planner evidence now share the same stage-2 axis shape
- the default planner benchmark path stays narrow unless explicitly widened
- BrowseComp gold can be benchmarked under both exact late interaction and
  clause-text reranking using the same explicit planner surface

This step does **not** prove:

- that clause-text should be enabled by default on all datasets
- that the current stage-2 sweep is complete
- that every public benchmark family already has the right text-sidecar source

It only proves that the benchmark substrate is now aligned enough to compare
planner behavior and planner evidence on the same explicit stage-2 dimension.
