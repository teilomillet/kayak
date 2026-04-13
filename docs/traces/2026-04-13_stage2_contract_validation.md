# 2026-04-13: stage-2 contract validation

## Why this step

The repository already had an implemented stage-2 boundary, but several public
surfaces still described it with legacy names:

- benchmark summaries still used `exact_stage_*`
- ceiling summaries still centered `reranker_kind`
- explain payloads exposed the new `stage2` profile, but coverage still mostly
  asserted the compatibility alias

That was not just a naming issue.

Reason:

- the repo is trying to make late interaction a first-class primitive
- a first-class primitive needs one explicit contract across engine, service,
  benchmarks, and Python
- compatibility aliases are useful for migration, but they are not the right
  primary language for new code

## Files changed

Primary implementation files:

- [kayak/planning/explain.mojo](../../kayak/planning/explain.mojo)
- [kayak/planning/json.mojo](../../kayak/planning/json.mojo)
- [kayak/benchmarks/stage_aware_json.mojo](../../kayak/benchmarks/stage_aware_json.mojo)
- [kayak/benchmarks/ceiling_comparison_json.mojo](../../kayak/benchmarks/ceiling_comparison_json.mojo)
- [kayak/benchmarks/__init__.mojo](../../kayak/benchmarks/__init__.mojo)

Coverage updates:

- [tests/test_stage_aware_benchmark_json.mojo](../../tests/test_stage_aware_benchmark_json.mojo)
- [tests/test_ceiling_comparison_json.mojo](../../tests/test_ceiling_comparison_json.mojo)
- [tests/test_single_core_scale_fixture.mojo](../../tests/test_single_core_scale_fixture.mojo)
- [tests/test_synthetic_hard_recall_fixture.mojo](../../tests/test_synthetic_hard_recall_fixture.mojo)
- [tests/test_collection_search_plan.mojo](../../tests/test_collection_search_plan.mojo)
- [tests/test_service_runtime.mojo](../../tests/test_service_runtime.mojo)
- [tests/test_service_json.mojo](../../tests/test_service_json.mojo)

## Verified decisions

### 1. Stage 2 is now the primary measured benchmark vocabulary

Implemented result:

- `StageAwareSearchSummary` now records:
  - `stage2_kind`
  - `stage2_family`
  - `stage2_requires_query_text`
  - `stage2_*` density fields
- legacy `exact_stage_*` keys are still emitted in JSON, but only as
  compatibility aliases mapped from the stage-2 values

Why this is justified:

- exact late interaction is one stage-2 operator, not the definition of stage 2
- the benchmark surface should match the explicit operator model already used by
  `SearchPlan`

### 2. Ceiling summaries now describe the real refinement operator

Implemented result:

- `CeilingComparisonSummary` now records:
  - `stage2_kind`
  - `stage2_family`
  - `stage2_requires_query_text`
- legacy `reranker_kind` is still emitted as a compatibility key

Why this is justified:

- `reranker_kind` is too narrow for operators like `noop_topk`
- the same summary surface can now describe:
  - identity top-k
  - exact late interaction
  - text-family refinement

### 3. Explain payload coverage now checks the new surface directly

Implemented result:

- `CollectionSearchExplain` carries both:
  - `stage2`
  - `exact_stage` compatibility alias
- tests now assert `stage2` directly on:
  - planner paths
  - hosted runtime path
  - debug JSON payloads

Why this is justified:

- otherwise a broken `stage2` field could be masked by a still-correct
  compatibility alias

## Commands run

All commands were executed from the clean worktree at:

- `/Users/teilomillet/Code/kayak/.worktrees/stage2-primitives`

Focused benchmark/summary checks:

```bash
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_stage_aware_benchmark_json.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_ceiling_comparison_json.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_single_core_scale_fixture.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_synthetic_hard_recall_fixture.mojo
```

Planner/service boundary checks:

```bash
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_collection_search_plan.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_service_json.mojo
_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__darwin_darwin pixi run --frozen mojo -I . tests/test_service_runtime.mojo
```

## Results

Observed pass counts:

- `tests/test_stage_aware_benchmark_json.mojo`: `2/2`
- `tests/test_ceiling_comparison_json.mojo`: `2/2`
- `tests/test_single_core_scale_fixture.mojo`: `2/2`
- `tests/test_synthetic_hard_recall_fixture.mojo`: `4/4`
- `tests/test_collection_search_plan.mojo`: `27/27`
- `tests/test_service_json.mojo`: `7/7`
- `tests/test_service_runtime.mojo`: `10/10`

Total verified in this step:

- `54` tests run
- `54` tests passed
- `0` tests failed

## What this does and does not prove

Verified:

- the explicit `stage2` contract is wired through:
  - planner explain output
  - benchmark summaries
  - ceiling summaries
  - hosted service debug output
- the renamed benchmark surface is not only syntactic; it survives the current
  end-to-end test coverage

Not verified here:

- any new retrieval-quality advantage from one stage-2 operator over another
- production latency on quiet hardware
- a broader migration of every remaining compatibility alias out of the repo

That narrower claim is intentional.

This step validates the contract boundary, not a new retrieval algorithm.
