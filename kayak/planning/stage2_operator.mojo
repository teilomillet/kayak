# Stage-2 refinement operator contract.

from std.collections import List


comptime STAGE2_OPERATOR_FAMILY_IDENTITY = "identity"
comptime STAGE2_OPERATOR_FAMILY_LATE_INTERACTION = "late_interaction"
comptime STAGE2_OPERATOR_FAMILY_TEXT = "text"

comptime STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT = "document_text"
comptime STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION = "late_interaction"


def stage2_family_for_kind(kind: String) raises -> String:
    if kind == "noop_topk":
        return STAGE2_OPERATOR_FAMILY_IDENTITY
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
    if kind == "exact_late_interaction":
        return [STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION]
    if kind == "clause_text":
        return [STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT]

    raise Error("unknown stage2 operator kind: " + kind)


def stage2_operator_requires_query_text(kind: String) raises -> Bool:
    if kind == "clause_text":
        return True
    if kind == "noop_topk" or kind == "exact_late_interaction":
        return False

    raise Error("unknown stage2 operator kind: " + kind)


def stage2_operator_is_exact_reference(kind: String) raises -> Bool:
    if kind == "exact_late_interaction":
        return True
    if kind == "noop_topk" or kind == "clause_text":
        return False

    raise Error("unknown stage2 operator kind: " + kind)


def compatibility_exact_stage_kind_for_stage2_operator(
    kind: String
) raises -> String:
    if kind == "exact_late_interaction":
        return kind
    if kind == "noop_topk" or kind == "clause_text":
        return "none"

    raise Error("unknown stage2 operator kind: " + kind)


def compatibility_reranker_kind_for_stage2_operator(
    kind: String
) raises -> String:
    if kind == "clause_text":
        return kind
    if kind == "noop_topk" or kind == "exact_late_interaction":
        return "none"

    raise Error("unknown stage2 operator kind: " + kind)


struct Stage2Operator(Copyable):
    var kind: String
    var family: String
    var required_artifact_families: List[String]
    var requires_query_text: Bool
    var is_exact_reference: Bool

    def __init__(out self):
        self.kind = "exact_late_interaction"
        self.family = STAGE2_OPERATOR_FAMILY_LATE_INTERACTION
        self.required_artifact_families = [STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION]
        self.requires_query_text = False
        self.is_exact_reference = True

    def __init__(out self, var kind: String) raises:
        self.kind = kind^
        self.family = stage2_family_for_kind(self.kind)
        self.required_artifact_families = (
            required_artifact_families_for_stage2_kind(self.kind)
        )
        self.requires_query_text = stage2_operator_requires_query_text(self.kind)
        self.is_exact_reference = stage2_operator_is_exact_reference(self.kind)

    def requires_artifact_family(self, family: String) -> Bool:
        for required_family in self.required_artifact_families:
            if required_family == family:
                return True

        return False


def noop_topk_stage2_operator() raises -> Stage2Operator:
    return Stage2Operator("noop_topk")


def exact_late_interaction_stage2_operator() raises -> Stage2Operator:
    return Stage2Operator("exact_late_interaction")


def clause_text_stage2_operator() raises -> Stage2Operator:
    return Stage2Operator("clause_text")
