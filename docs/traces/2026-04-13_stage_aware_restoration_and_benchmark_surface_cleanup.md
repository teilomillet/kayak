# Stage-Aware Restoration And Benchmark Surface Cleanup

Date: `2026-04-13`
Status: `verified in worktree`

## Claim

Two changes were needed together:

1. benchmark setup surfaces should describe refinement in explicit stage terms,
   not through the legacy combined `Stage2Operator` view
2. pooled stage-1 benchmarking should distinguish stage-1 candidate quality from
   stage-2 exact late-interaction restoration quality

Reason:

- without the first change, the code still encourages readers to think in old
  combined-stage language even when runtime semantics are already explicit
- without the second change, pooled retrieval gains can be mistaken for
  reference-faithful late-interaction gains

## Code Checked

- [kayak/planning/search_plan.mojo](../../kayak/planning/search_plan.mojo)
- [kayak/benchmarks/planner_benchmark_json.mojo](../../kayak/benchmarks/planner_benchmark_json.mojo)
- [kayak/benchmarks/planner_benchmark_runner.mojo](../../kayak/benchmarks/planner_benchmark_runner.mojo)
- [kayak/benchmarks/planner_evidence_json.mojo](../../kayak/benchmarks/planner_evidence_json.mojo)
- [kayak/benchmarks/query_text_support.mojo](../../kayak/benchmarks/query_text_support.mojo)
- [kayak/benchmarks/search_plan_semantics_json.mojo](../../kayak/benchmarks/search_plan_semantics_json.mojo)
- [kayak/benchmarks/token_pooling_common.mojo](../../kayak/benchmarks/token_pooling_common.mojo)
- [kayak/benchmarks/token_pooling_stage_aware_json.mojo](../../kayak/benchmarks/token_pooling_stage_aware_json.mojo)
- [benchmarks/real_subset_planner_benchmark_smoke.mojo](../../benchmarks/real_subset_planner_benchmark_smoke.mojo)
- [benchmarks/browsecomp_plus_gold_token_pooling_stage_aware.mojo](../../benchmarks/browsecomp_plus_gold_token_pooling_stage_aware.mojo)

## What Changed

### 1. Benchmark setup surfaces now use explicit stage components

Verified code changes:

- `build_planner_benchmark_summary(...)` now takes
  `stage2_reference_operator` and `stage3_verifier`
- `build_planner_evidence_summary(...)` now takes
  `stage2_reference_operator` and `stage3_verifier`
- benchmark runner setup now builds explicit refinement setups instead of
  sweeping `Stage2Operator`
- query-text support now keys off `plan.stage3_verifier.requires_query_text`,
  which matches the real verifier dependency

Why this is justified:

- the runtime already executes stage 2 and stage 3 separately
- leaving benchmark setup in the old combined language would preserve a false
  mental model even after runtime semantics were fixed

### 2. Added a stage-aware token-pooling benchmark

Verified new benchmark pieces:

- shared helpers moved into
  [token_pooling_common.mojo](../../kayak/benchmarks/token_pooling_common.mojo)
- new stage-aware benchmark logic added in
  [token_pooling_stage_aware_json.mojo](../../kayak/benchmarks/token_pooling_stage_aware_json.mojo)
- new entrypoint added in
  [browsecomp_plus_gold_token_pooling_stage_aware.mojo](../../benchmarks/browsecomp_plus_gold_token_pooling_stage_aware.mojo)

Its contract is:

- stage 1: exact search over a pooled document index
- stage 2: exact late-interaction scoring over the original unpooled documents,
  restricted to the stage-1 candidate window

Why this is justified:

- it separates "pooled search found useful candidates" from
  "exact late interaction agrees with the pooled ranking"
- that separation is necessary if token pooling is used as a stage-1 retrieval
  device rather than as a replacement for exact reference semantics

## Validation

### Focused tests

Commands run:

```bash
pixi run mojo -I . tests/test_planner_benchmark_json.mojo
pixi run mojo -I . tests/test_planner_evidence_json.mojo
pixi run mojo -I . tests/test_token_pooling_json.mojo
pixi run mojo -I . tests/test_token_pooling_stage_aware_json.mojo
PYTHONPATH=python pixi run python -m unittest -q python/tests/test_stage_semantic_guardrails.py
```

Observed result:

- all listed tests passed locally in the `stage-aware-restoration` worktree

### Entry-point execution check

Command run:

```bash
pixi run mojo -I . benchmarks/real_subset_planner_benchmark_smoke.mojo
```

Observed result:

- completed successfully
- wrote `.cache/kayak/public_planner_benchmark_smoke.json`

JSON spot-check:

- `reference_scoring_semantics_kind=exact_late_interaction`
- `stage2_reference_kind=exact_late_interaction`
- `stage3_verifier_kind=none`
- `compatibility_stage2_kind=exact_late_interaction`

Interpretation:

- the executable benchmark surface now emits explicit semantic fields first
- the old combined stage view remains present only as a compatibility field

### Stage-aware token-pooling benchmark

Command run:

```bash
pixi run bench_browsecomp_plus_gold_token_pooling_stage_aware_raw
```

Verified findings from
`.cache/kayak/browsecomp_plus_gold_token_pooling_stage_aware.json`:

- hierarchical pooling, factor `2`, `candidate_k=10`
  - `mean_stage1_candidate_recall_at_final_k = 0.975`
  - `mean_restored_reference_recall_at_final_k = 0.975`
- hierarchical pooling, factor `2`, `candidate_k=20`
  - `mean_stage1_candidate_recall_at_final_k = 1.0`
  - `mean_restored_reference_recall_at_final_k = 1.0`
- hierarchical pooling, factor `3`, `candidate_k=10`
  - stage-1 pooled nDCG was higher than the exact baseline
  - restored exact nDCG dropped back toward the exact reference
- hierarchical pooling, factor `3`, `candidate_k=20`
  - restored recall returned to `1.0`
  - restored nDCG returned to the exact-reference value on this slice

Interpretation:

- exact stage 2 only restores what survives the stage-1 candidate window
- pooled-only gains are not automatically exact-reference gains
- candidate recall and restored exact faithfulness are different axes and need
  to be reported separately

## Additional Guardrail

Added:

- [python/tests/test_stage_semantic_guardrails.py](../../python/tests/test_stage_semantic_guardrails.py)

What it checks:

- legacy combined stage naming is only allowed in a small explicit
  compatibility allowlist
- new production files that reintroduce combined-stage language outside that
  boundary fail the guardrail

Why this is justified:

- the current risk is semantic backsliding, not missing runtime structure
- a lightweight allowlist test is a cheap way to keep new code in the explicit
  stage-1 / stage-2 / stage-3 language

## Open Questions

- the compatibility layer still exists in service and Python SDK surfaces, so
  downstream consumers can still choose the old language for now
- the smoke planner benchmark emitted a Python `resource_tracker` leaked
  semaphore warning at shutdown; that did not change exit status, but it should
  not be interpreted as evidence that the dependency stack is perfectly clean
