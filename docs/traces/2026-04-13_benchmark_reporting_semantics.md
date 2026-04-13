# Benchmark Reporting Semantics

## Claim

Benchmark and evidence JSON should describe retrieval semantics in the same
terms as `SearchPlan`, rather than collapsing everything into legacy combined
fields like `stage2_kind`.

## Why

The planner already distinguishes:

- stage-1 candidate generation
- reference scoring semantics
- stage-2 reference execution
- stage-3 verification

But several benchmark summaries still reported only one combined stage-2 label.
That made WARP-style centroid engines, proxy stages, and GEM-style graph stages
harder to compare in retrieval terms, and it also hid the difference between
the stage-2 reference window and the full exact oracle.

## Verified local findings

- `kayak/planning/json.mojo` and `kayak/service/json.mojo` already emitted the
  explicit semantic split.
- `kayak/benchmarks/stage_aware_json.mojo`,
  `planner_benchmark_json.mojo`,
  `planner_evidence_json.mojo`, and
  `ceiling_comparison_json.mojo`
  still centered legacy combined stage-2 vocabulary before this change.
- `stage_aware_json.mojo` also aliased exact-oracle density fields to the
  stage-2 reference stage, which was semantically wrong.

## Changes

- Faithfulness frontier summaries now store the real `CandidateGenerator` and
  emit stage-1 semantic provenance directly.
- Planner benchmark, planner evidence, stage-aware, and ceiling summaries now
  store the real `SearchPlan` and emit:
  - `reference_scoring_semantics_*`
  - `stage2_reference_*`
  - `stage3_verifier_*`
- Compatibility fields are still present, but only as explicit
  `compatibility_*` keys.
- Stage-aware summaries now report:
  - `stage2_reference_*` materialization and density
  - `stage3_verifier_*` materialization and density
  - `exact_oracle_*` density separately

## Validation

Verified with:

```bash
pixi run mojo -I . tests/test_faithfulness_frontier_json.mojo
pixi run mojo -I . tests/test_planner_benchmark_json.mojo
pixi run mojo -I . tests/test_planner_evidence_json.mojo
pixi run mojo -I . tests/test_stage_aware_benchmark_json.mojo
pixi run mojo -I . tests/test_ceiling_comparison_json.mojo
pixi run mojo -I . tests/test_single_core_scale_fixture.mojo
pixi run mojo -I . tests/test_synthetic_hard_recall_fixture.mojo
```

All passed locally after the change.
