# Compatibility view over the legacy combined stage-2 operator surface.

from std.collections import List

from .reference_scoring_semantics import (
    ReferenceScoringSemantics,
    exact_late_interaction_reference_scoring_semantics,
)
from .stage2_reference_operator import (
    STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY,
    STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION,
    STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
    Stage2ReferenceOperator,
    exact_late_interaction_stage2_reference_operator,
    noop_topk_stage2_reference_operator,
)
from .stage3_verifier_operator import (
    STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
    STAGE3_VERIFIER_FAMILY_TEXT,
    Stage3VerifierOperator,
    clause_text_stage3_verifier_operator,
    none_stage3_verifier_operator,
)


comptime STAGE2_OPERATOR_FAMILY_IDENTITY = STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY
comptime STAGE2_OPERATOR_FAMILY_HYBRID = "hybrid"
comptime STAGE2_OPERATOR_FAMILY_LATE_INTERACTION = (
    STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION
)
comptime STAGE2_OPERATOR_FAMILY_TEXT = STAGE3_VERIFIER_FAMILY_TEXT

comptime STAGE2_EXECUTION_KIND_NOOP_TOPK = "noop_topk"
comptime STAGE2_EXECUTION_KIND_EXACT_LATE_INTERACTION = "exact_late_interaction"
comptime STAGE2_EXECUTION_KIND_EXACT_LATE_INTERACTION_CLAUSE_TEXT = (
    "exact_late_interaction_clause_text"
)
comptime STAGE2_EXECUTION_KIND_CLAUSE_TEXT = "clause_text"

comptime STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT = (
    STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT
)
comptime STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION = (
    STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION
)


def stage2_family_for_kind(kind: String) raises -> String:
    if kind == "noop_topk":
        return STAGE2_OPERATOR_FAMILY_IDENTITY
    if kind == "exact_late_interaction_clause_text":
        return STAGE2_OPERATOR_FAMILY_HYBRID
    if kind == "exact_late_interaction":
        return STAGE2_OPERATOR_FAMILY_LATE_INTERACTION
    if kind == "clause_text":
        return STAGE2_OPERATOR_FAMILY_TEXT

    raise Error("unknown stage2 operator kind: " + kind)


def required_artifact_families_for_stage2_kind(
    kind: String
) raises -> List[String]:
    if kind == "noop_topk":
        return List[String]()
    if kind == "exact_late_interaction_clause_text":
        return [
            STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION,
            STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
        ]
    if kind == "exact_late_interaction":
        return [STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION]
    if kind == "clause_text":
        return [STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT]

    raise Error("unknown stage2 operator kind: " + kind)


def stage2_operator_requires_query_text(kind: String) raises -> Bool:
    if kind == "exact_late_interaction_clause_text" or kind == "clause_text":
        return True
    if kind == "noop_topk" or kind == "exact_late_interaction":
        return False

    raise Error("unknown stage2 operator kind: " + kind)


def stage2_operator_is_exact_reference(kind: String) raises -> Bool:
    if kind == "exact_late_interaction":
        return True
    if (
        kind == "noop_topk"
        or kind == "clause_text"
        or kind == "exact_late_interaction_clause_text"
    ):
        return False

    raise Error("unknown stage2 operator kind: " + kind)


def stage2_execution_kind_for_operator(kind: String) raises -> String:
    if kind == "noop_topk":
        return STAGE2_EXECUTION_KIND_NOOP_TOPK
    if kind == "exact_late_interaction":
        return STAGE2_EXECUTION_KIND_EXACT_LATE_INTERACTION
    if kind == "exact_late_interaction_clause_text":
        return STAGE2_EXECUTION_KIND_EXACT_LATE_INTERACTION_CLAUSE_TEXT
    if kind == "clause_text":
        return STAGE2_EXECUTION_KIND_CLAUSE_TEXT

    raise Error("unknown stage2 operator kind: " + kind)


def reference_scoring_semantics_for_stage2_operator_kind(
    kind: String
) raises -> ReferenceScoringSemantics:
    _ = stage2_family_for_kind(kind)
    return exact_late_interaction_reference_scoring_semantics()


def stage2_reference_operator_for_stage2_operator_kind(
    kind: String
) raises -> Stage2ReferenceOperator:
    if kind == "noop_topk" or kind == "clause_text":
        return noop_topk_stage2_reference_operator()
    if kind == "exact_late_interaction" or kind == "exact_late_interaction_clause_text":
        return exact_late_interaction_stage2_reference_operator()

    raise Error("unknown stage2 operator kind: " + kind)


def stage3_verifier_for_stage2_operator_kind(
    kind: String
) raises -> Stage3VerifierOperator:
    if kind == "noop_topk" or kind == "exact_late_interaction":
        return none_stage3_verifier_operator()
    if kind == "clause_text" or kind == "exact_late_interaction_clause_text":
        return clause_text_stage3_verifier_operator()

    raise Error("unknown stage2 operator kind: " + kind)


def combined_stage2_operator_kind(
    read stage2_reference_operator: Stage2ReferenceOperator,
    read stage3_verifier: Stage3VerifierOperator,
) raises -> String:
    if stage2_reference_operator.kind == "noop_topk":
        if stage3_verifier.kind == "none":
            return "noop_topk"
        if stage3_verifier.kind == "clause_text":
            return "clause_text"
    elif stage2_reference_operator.kind == "exact_late_interaction":
        if stage3_verifier.kind == "none":
            return "exact_late_interaction"
        if stage3_verifier.kind == "clause_text":
            return "exact_late_interaction_clause_text"

    raise Error(
        "unsupported compatibility stage-2 composition: "
        + stage2_reference_operator.kind
        + " + "
        + stage3_verifier.kind
    )


def compatibility_exact_stage_kind_for_stage2_operator(
    kind: String
) raises -> String:
    if kind == "exact_late_interaction":
        return "exact_late_interaction"
    if (
        kind == "noop_topk"
        or kind == "clause_text"
        or kind == "exact_late_interaction_clause_text"
    ):
        return "none"

    raise Error("unknown stage2 operator kind: " + kind)


def compatibility_reranker_kind_for_stage2_operator(
    kind: String
) raises -> String:
    if kind == "exact_late_interaction_clause_text":
        return "clause_text"
    if kind == "clause_text":
        return kind
    if kind == "noop_topk" or kind == "exact_late_interaction":
        return "none"

    raise Error("unknown stage2 operator kind: " + kind)


struct Stage2Operator(Copyable):
    var kind: String
    var family: String
    var execution_kind: String
    var required_artifact_families: List[String]
    var requires_query_text: Bool
    var is_exact_reference: Bool
    var compatibility_exact_stage_kind: String
    var compatibility_reranker_kind: String

    def __init__(out self):
        self.kind = "exact_late_interaction"
        self.family = STAGE2_OPERATOR_FAMILY_LATE_INTERACTION
        self.execution_kind = STAGE2_EXECUTION_KIND_EXACT_LATE_INTERACTION
        self.required_artifact_families = [STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION]
        self.requires_query_text = False
        self.is_exact_reference = True
        self.compatibility_exact_stage_kind = "exact_late_interaction"
        self.compatibility_reranker_kind = "none"

    def __init__(out self, var kind: String) raises:
        self.kind = kind^
        self.family = stage2_family_for_kind(self.kind)
        self.execution_kind = stage2_execution_kind_for_operator(self.kind)
        self.required_artifact_families = (
            required_artifact_families_for_stage2_kind(self.kind)
        )
        self.requires_query_text = stage2_operator_requires_query_text(self.kind)
        self.is_exact_reference = stage2_operator_is_exact_reference(self.kind)
        self.compatibility_exact_stage_kind = (
            compatibility_exact_stage_kind_for_stage2_operator(self.kind)
        )
        self.compatibility_reranker_kind = (
            compatibility_reranker_kind_for_stage2_operator(self.kind)
        )

    def requires_artifact_family(self, family: String) -> Bool:
        for required_family in self.required_artifact_families:
            if required_family == family:
                return True

        return False


def noop_topk_stage2_operator() raises -> Stage2Operator:
    return Stage2Operator("noop_topk")


def exact_late_interaction_stage2_operator() raises -> Stage2Operator:
    return Stage2Operator("exact_late_interaction")


def exact_late_interaction_clause_text_stage2_operator() raises -> Stage2Operator:
    return Stage2Operator("exact_late_interaction_clause_text")


def clause_text_stage2_operator() raises -> Stage2Operator:
    return Stage2Operator("clause_text")


def stage2_operator_for_components(
    read stage2_reference_operator: Stage2ReferenceOperator,
    read stage3_verifier: Stage3VerifierOperator,
) raises -> Stage2Operator:
    return Stage2Operator(
        combined_stage2_operator_kind(stage2_reference_operator, stage3_verifier)
    )
