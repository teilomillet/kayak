# 2026-04-13: stage semantic guardrails and compatibility surface

## Claim

The stage-semantic guardrails are now first-class verification surfaces, and
the planner/service compatibility view is derived from one canonical helper
instead of being reassembled independently in each consumer.

## Why This Change Was Needed

The repo already had good explicit stage structure:
- `reference_scoring_semantics`
- `stage2_reference_operator`
- `stage3_verifier`

But the legacy compatibility view still leaked through several places:
- `SearchPlan` cached compatibility fields directly
- service JSON re-read those fields ad hoc
- explain JSON re-read those fields ad hoc
- service contract equality re-compared those fields ad hoc
- the Python bridge mirrored the same pattern

That shape was readable, but it was still a semantic drift seam.

## What Changed

### 1. Stage guardrails now have a named gate

`pyproject.toml` now includes:
- `test_stage_semantic_guardrails`
- `test_stage1_semantic_guardrails`
- `test_semantic_guardrails`

Reason:
- the repo inventory and stage guardrails should be runnable as one fast
  contract gate rather than scattered across unrelated commands

### 2. Planner-owned compatibility semantics

`kayak/planning/search_plan.mojo` now owns:
- `SearchPlanCompatibilitySemantics`
- `search_plan_compatibility_semantics_for_components(...)`
- `search_plan_compatibility_semantics(...)`

Reason:
- compatibility naming should be derived from explicit planner components in one
  place
- service and explain code should consume that planner-owned view instead of
  reconstructing it

Follow-on tightening in the same tranche:
- `SearchPlan` no longer stores cached `exact_stage_kind` or `reranker_kind`
- the compatibility projection is derived when needed

### 3. Service and benchmark consumers now read the helper

Updated consumers:
- `kayak/benchmarks/search_plan_semantics_json.mojo`
- `kayak/planning/json.mojo`
- `kayak/service/json.mojo`
- `kayak/service/search_contracts.mojo`

The narrower claim is:
- these files no longer each own their own compatibility-stage interpretation

### 4. Python bridge mirrors the same cleanup

`python/kayak_bridge/search_plan.py` now has the same planner-owned
compatibility derivation idea on the local SDK side.

Reason:
- the Python SDK should not preserve a parallel drift seam after the Mojo side
  is cleaned up

### 5. Compatibility quarantine is now token-specific

The Python semantic guardrail no longer uses one broad file allowlist.

It now scopes ownership per token:
- `Stage2Operator`
- `stage2_operator_kind`
- `search_plan_with_stage2_operator`
- `exact_stage_kind`
- `reranker_kind`

Reason:
- broad substring quarantine was too coarse
- helper names such as `...for_stage2_operator_kind` are not the same thing as
  a real compatibility surface

## Verification

Commands run:

```bash
pixi run test_semantic_guardrails
pixi run mojo -I . tests/test_stage2_operator_contract.mojo
pixi run mojo -I . tests/test_service_contracts.mojo
pixi run mojo -I . tests/test_service_json.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run test_python_api
```

Observed results:
- `test_semantic_guardrails`: passed
- `tests/test_stage2_operator_contract.mojo`: passed
- `tests/test_service_contracts.mojo`: passed
- `tests/test_service_json.mojo`: passed
- `tests/test_collection_search_plan.mojo`: passed
- `test_python_api`: passed, `45` tests

## Residual Risk

This does not remove the public compatibility fields yet.

It does something narrower and more useful first:
- keeps compatibility support intact
- makes its derivation explicit and centralized
- makes future removal of compatibility fields less risky because fewer files
  own their meaning directly
