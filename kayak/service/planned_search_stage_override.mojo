# Helpers for validating and resolving planned-search stage overrides while
# preserving the explicit multi-stage search-plan semantics.

from kayak.planning import (
    SearchPlan,
    Stage2Operator,
    Stage2ReferenceOperator,
    Stage3VerifierOperator,
    stage2_operator_for_components,
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
    stage2_operator_kind: String,
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) -> Bool:
    return (
        stage2_operator_kind.byte_length() > 0
        or planned_search_has_component_stage_override(
            stage2_reference_kind,
            stage3_verifier_kind,
        )
    )


def require_planned_search_stage_override_not_mixed(
    stage2_operator_kind: String,
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) raises:
    if (
        stage2_operator_kind.byte_length() > 0
        and planned_search_has_component_stage_override(
            stage2_reference_kind,
            stage3_verifier_kind,
        )
    ):
        raise Error(
            "planned search request cannot mix compatibility stage2_operator_kind with explicit stage2_reference_kind or stage3_verifier_kind"
        )


def require_valid_planned_search_stage_override(
    query_text: String,
    stage2_operator_kind: String,
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) raises:
    require_planned_search_stage_override_not_mixed(
        stage2_operator_kind,
        stage2_reference_kind,
        stage3_verifier_kind,
    )

    if stage2_operator_kind.byte_length() > 0:
        var stage2_operator = Stage2Operator(stage2_operator_kind.copy())
        if (
            stage2_operator.requires_query_text
            and query_text.byte_length() == 0
        ):
            raise Error(
                "planned stage2 operator "
                + stage2_operator.kind
                + " requires non-empty query_text"
            )
        return

    var stage2_reference_operator = Stage2ReferenceOperator()
    var stage3_verifier = Stage3VerifierOperator()
    var has_stage2_reference_override = False
    var has_stage3_verifier_override = False

    if stage2_reference_kind.byte_length() > 0:
        stage2_reference_operator = Stage2ReferenceOperator(
            stage2_reference_kind.copy()
        )
        has_stage2_reference_override = True

    if stage3_verifier_kind.byte_length() > 0:
        stage3_verifier = Stage3VerifierOperator(stage3_verifier_kind.copy())
        has_stage3_verifier_override = True
        if (
            stage3_verifier.requires_query_text
            and query_text.byte_length() == 0
        ):
            raise Error(
                "planned stage3 verifier "
                + stage3_verifier.kind
                + " requires non-empty query_text"
            )

    if has_stage2_reference_override and has_stage3_verifier_override:
        _ = stage2_operator_for_components(
            stage2_reference_operator,
            stage3_verifier,
        )


def search_plan_with_planned_search_stage_override(
    read plan: SearchPlan,
    stage2_operator_kind: String,
    stage2_reference_kind: String,
    stage3_verifier_kind: String,
) raises -> SearchPlan:
    require_planned_search_stage_override_not_mixed(
        stage2_operator_kind,
        stage2_reference_kind,
        stage3_verifier_kind,
    )

    if stage2_operator_kind.byte_length() > 0:
        return SearchPlan(
            plan.candidate_generator,
            plan.candidate_budget,
            plan.faithfulness_policy,
            Stage2Operator(stage2_operator_kind.copy()),
        )

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
