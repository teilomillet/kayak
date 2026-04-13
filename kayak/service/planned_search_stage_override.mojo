# Helpers for validating and resolving planned-search stage overrides while
# preserving the explicit multi-stage search-plan semantics.

from kayak.planning import (
    SearchPlan,
    Stage2ReferenceOperator,
    Stage3VerifierOperator,
)


def planned_search_has_component_stage_override(
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) -> Bool:
    return (
        stage2_reference_kind.byte_length() > 0
        or stage3_verifier_kind.byte_length() > 0
    )


def planned_search_has_stage_override(
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) -> Bool:
    return planned_search_has_component_stage_override(
        stage2_reference_kind,
        stage3_verifier_kind,
    )


def require_valid_planned_search_stage_override(
    query_text: String,
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) raises:
    if stage2_reference_kind.byte_length() > 0:
        _ = Stage2ReferenceOperator(stage2_reference_kind.copy())

    if stage3_verifier_kind.byte_length() > 0:
        var stage3_verifier = Stage3VerifierOperator(stage3_verifier_kind.copy())
        if (
            stage3_verifier.requires_query_text
            and query_text.byte_length() == 0
        ):
            raise Error(
                "planned stage3 verifier "
                + stage3_verifier.kind
                + " requires non-empty query_text"
            )


def search_plan_with_planned_search_stage_override(
    read plan: SearchPlan,
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) raises -> SearchPlan:
    if not planned_search_has_component_stage_override(
        stage2_reference_kind,
        stage3_verifier_kind,
    ):
        return plan.copy()

    var stage2_reference_operator = plan.stage2_reference_operator.copy()
    if stage2_reference_kind.byte_length() > 0:
        stage2_reference_operator = Stage2ReferenceOperator(
            stage2_reference_kind.copy()
        )

    var stage3_verifier = plan.stage3_verifier.copy()
    if stage3_verifier_kind.byte_length() > 0:
        stage3_verifier = Stage3VerifierOperator(stage3_verifier_kind.copy())

    return SearchPlan(
        plan.candidate_generator,
        plan.candidate_budget,
        plan.faithfulness_policy,
        plan.reference_scoring_semantics,
        stage2_reference_operator,
        stage3_verifier,
    )
