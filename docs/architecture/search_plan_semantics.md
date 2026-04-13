# Search Plan Semantics

Status: current repository contract
Date: `2026-04-13`

This document describes the canonical search-plan semantics that the repo now
enforces in code.

## Canonical plan shape

`SearchPlan` is an explicit multi-stage contract with these fields:

- `candidate_generator`
- `candidate_budget`
- `reference_scoring_semantics`
- `stage2_reference_operator`
- `stage3_verifier`
- `faithfulness_policy`

Reason:

- stage-1 retrieval, exact reference scoring, and text verification have
  different semantics and different artifact requirements
- the repo now treats those distinctions as first-class state instead of
  compressing them into one combined stage name

## Current explicit stage semantics

The current supported refinement pieces are:

- reference scoring semantics:
  - `exact_late_interaction`
- stage-2 reference operators:
  - `noop_topk`
  - `exact_late_interaction`
- stage-3 verifiers:
  - `none`
  - `clause_text`

That contract is reflected in:

- [kayak/planning/search_plan.mojo](../../kayak/planning/search_plan.mojo)
- [kayak/planning/stage2_reference_operator.mojo](../../kayak/planning/stage2_reference_operator.mojo)
- [kayak/planning/stage3_verifier_operator.mojo](../../kayak/planning/stage3_verifier_operator.mojo)
- [kayak/service/search_contracts.mojo](../../kayak/service/search_contracts.mojo)

## Planner and service implications

`PlannedSearchRequest` overrides stage behavior explicitly:

- `stage2_reference_kind`
- `stage3_verifier_kind`

Reason:

- overriding stage-2 reference scoring is a different operation from enabling a
  stage-3 text verifier
- filter-aware planning and exact fallback behavior need those ownership
  boundaries to remain visible

## Guardrails

The repo now enforces this contract mechanically in production source:

- no combined-stage compatibility module remains in `kayak/` or
  `python/kayak_bridge/`
- production code must not reintroduce legacy combined-stage naming
- deleted compatibility modules are checked to stay deleted

The primary regression gate is:

- `pixi run test_semantic_guardrails`

## Verification

This explicit contract should be validated against the plan, service, and
Python SDK tests whenever it changes. The most relevant checks are:

- `tests/test_stage_refinement_contract.mojo`
- `tests/test_search_planner.mojo`
- `tests/test_service_contracts.mojo`
- `tests/test_collection_search_plan.mojo`
- `tests/test_service_runtime.mojo`
- `python/tests/test_search_plan_api.py`
- `python/tests/test_public_api_contract.py`
- `python/tests/test_stage_semantic_guardrails.py`
