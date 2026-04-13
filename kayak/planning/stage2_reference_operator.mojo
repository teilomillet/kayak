# Stage-2 reference operators realize the reference semantics over a bounded
# candidate window when stage-1 did not already do so.

from std.collections import List


comptime STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY = "identity"
comptime STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION = "late_interaction"
comptime STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION = "late_interaction"


def stage2_reference_family_for_kind(kind: String) raises -> String:
    if kind == "noop_topk":
        return STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY
    if kind == "exact_late_interaction":
        return STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION

    raise Error("unknown stage-2 reference operator kind: " + kind)


def required_artifact_families_for_stage2_reference_kind(
    kind: String
) raises -> List[String]:
    if kind == "noop_topk":
        return List[String]()
    if kind == "exact_late_interaction":
        return [STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION]

    raise Error("unknown stage-2 reference operator kind: " + kind)


def stage2_reference_operator_executes_reference_scoring(
    kind: String
) raises -> Bool:
    if kind == "exact_late_interaction":
        return True
    if kind == "noop_topk":
        return False

    raise Error("unknown stage-2 reference operator kind: " + kind)


struct Stage2ReferenceOperator(Copyable):
    var kind: String
    var family: String
    var required_artifact_families: List[String]
    var executes_reference_scoring: Bool

    def __init__(out self):
        self.kind = "exact_late_interaction"
        self.family = STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION
        self.required_artifact_families = [
            STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION
        ]
        self.executes_reference_scoring = True

    def __init__(out self, var kind: String) raises:
        self.kind = kind^
        self.family = stage2_reference_family_for_kind(self.kind)
        self.required_artifact_families = (
            required_artifact_families_for_stage2_reference_kind(self.kind)
        )
        self.executes_reference_scoring = (
            stage2_reference_operator_executes_reference_scoring(self.kind)
        )

    def requires_artifact_family(self, family: String) -> Bool:
        for required_family in self.required_artifact_families:
            if required_family == family:
                return True

        return False


def noop_topk_stage2_reference_operator() raises -> Stage2ReferenceOperator:
    return Stage2ReferenceOperator("noop_topk")


def exact_late_interaction_stage2_reference_operator(
) -> Stage2ReferenceOperator:
    return Stage2ReferenceOperator()
