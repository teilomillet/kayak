# Helpers for validating and resolving planned-search stage overrides while
# preserving the explicit multi-stage search-plan semantics.

from kayak.planning import (
    SearchPlan,
    Stage2ReferenceOperator,
    Stage3VerifierOperator,
    noop_topk_stage2_reference_operator,
    none_stage3_verifier_operator,
)


struct PlannedSearchStageOverride(Copyable):
    var has_stage2_reference_override: Bool
    var stage2_reference_operator: Stage2ReferenceOperator
    var has_stage3_verifier_override: Bool
    var stage3_verifier: Stage3VerifierOperator

    def __init__(out self) raises:
        self.has_stage2_reference_override = False
        self.stage2_reference_operator = noop_topk_stage2_reference_operator()
        self.has_stage3_verifier_override = False
        self.stage3_verifier = none_stage3_verifier_operator()

    def __init__(
        out self,
        query_text: String,
        var stage2_reference_kind: String,
        var stage3_verifier_kind: String,
    ) raises:
        self = PlannedSearchStageOverride()

        if stage2_reference_kind.byte_length() > 0:
            self.has_stage2_reference_override = True
            self.stage2_reference_operator = Stage2ReferenceOperator(
                stage2_reference_kind^
            )

        if stage3_verifier_kind.byte_length() > 0:
            self.has_stage3_verifier_override = True
            self.stage3_verifier = Stage3VerifierOperator(stage3_verifier_kind^)
            if (
                self.stage3_verifier.requires_query_text
                and query_text.byte_length() == 0
            ):
                raise Error(
                    "planned stage3 verifier "
                    + self.stage3_verifier.kind
                    + " requires non-empty query_text"
                )


def planned_search_has_component_stage_override(
    read stage_override: PlannedSearchStageOverride
) -> Bool:
    return (
        stage_override.has_stage2_reference_override
        or stage_override.has_stage3_verifier_override
    )


def planned_search_has_stage_override(
    read stage_override: PlannedSearchStageOverride
) -> Bool:
    return planned_search_has_component_stage_override(stage_override)


def search_plan_with_planned_search_stage_override(
    read plan: SearchPlan,
    read stage_override: PlannedSearchStageOverride,
) raises -> SearchPlan:
    if not planned_search_has_component_stage_override(stage_override):
        return plan.copy()

    var stage2_reference_operator = plan.stage2_reference_operator.copy()
    if stage_override.has_stage2_reference_override:
        stage2_reference_operator = stage_override.stage2_reference_operator.copy()

    var stage3_verifier = plan.stage3_verifier.copy()
    if stage_override.has_stage3_verifier_override:
        stage3_verifier = stage_override.stage3_verifier.copy()

    return SearchPlan(
        plan.candidate_generator,
        plan.candidate_budget,
        plan.faithfulness_policy,
        plan.reference_scoring_semantics,
        stage2_reference_operator,
        stage3_verifier,
    )
