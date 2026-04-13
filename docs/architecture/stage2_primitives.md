# Stage-2 Primitives

Status: current repository contract
Date: `2026-04-13`

This document records the stage-aware refinement primitives that are actually
implemented now.

## Why the split exists

Late-interaction refinement in this repo is no longer modeled as one combined
operator because three concerns need independent ownership:

- reference scoring semantics describe what truth-preserving scoring means
- stage-2 reference operators describe whether the candidate window executes
  that scoring or just forwards hits
- stage-3 verifiers describe any additional query-text or document-text
  verification work

Reason:

- exact late interaction and clause-text verification are not the same kind of
  operation
- storage, required artifacts, and latency analysis depend on which stage owns
  each action

## Implemented primitives

Reference scoring semantics:

- `exact_late_interaction`

Stage-2 reference operators:

- `noop_topk`
- `exact_late_interaction`

Stage-3 verifiers:

- `none`
- `clause_text`

These are defined in:

- [kayak/planning/reference_scoring_semantics.mojo](../../kayak/planning/reference_scoring_semantics.mojo)
- [kayak/planning/stage2_reference_operator.mojo](../../kayak/planning/stage2_reference_operator.mojo)
- [kayak/planning/stage3_verifier_operator.mojo](../../kayak/planning/stage3_verifier_operator.mojo)

## Search plan integration

`SearchPlan` stores the three explicit pieces directly.

That means:

- exact late interaction remains the correctness anchor
- clause-text verification stays explicit instead of hiding inside a combined
  stage alias
- service and benchmark JSON can report the actual staged contract instead of
  projecting it back into a synthetic combined name

## Python SDK implication

The Python SDK follows the same boundary:

- `Stage2ReferenceOperator`
- `Stage3VerifierOperator`
- `ReferenceScoringSemantics`

The public plan builders accept explicit `stage2_reference_operator` and
`stage3_verifier` overrides only.

## Verification

Relevant checks:

- `tests/test_stage_refinement_contract.mojo`
- `tests/test_collection_search_plan.mojo`
- `tests/test_service_runtime.mojo`
- `python/tests/test_search_plan_api.py`
- `python/tests/test_public_api_contract.py`
