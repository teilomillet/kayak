# Reference scoring semantics name the truth scoring contract independent of
# which execution stage realizes it.

from std.collections import List


comptime REFERENCE_SCORING_SEMANTICS_FAMILY_LATE_INTERACTION = "late_interaction"
comptime REFERENCE_SCORING_REQUIRED_ARTIFACT_LATE_INTERACTION = "late_interaction"
comptime REFERENCE_SCORING_SCORE_KIND_EXACT = "exact_score"


def reference_scoring_family_for_kind(kind: String) raises -> String:
    if kind == "exact_late_interaction":
        return REFERENCE_SCORING_SEMANTICS_FAMILY_LATE_INTERACTION

    raise Error("unknown reference scoring semantics kind: " + kind)


def required_artifact_families_for_reference_scoring_kind(
    kind: String
) raises -> List[String]:
    if kind == "exact_late_interaction":
        return [REFERENCE_SCORING_REQUIRED_ARTIFACT_LATE_INTERACTION]

    raise Error("unknown reference scoring semantics kind: " + kind)


def score_kind_for_reference_scoring_kind(kind: String) raises -> String:
    if kind == "exact_late_interaction":
        return REFERENCE_SCORING_SCORE_KIND_EXACT

    raise Error("unknown reference scoring semantics kind: " + kind)


struct ReferenceScoringSemantics(Copyable):
    var kind: String
    var family: String
    var required_artifact_families: List[String]
    var score_kind: String

    def __init__(out self):
        self.kind = "exact_late_interaction"
        self.family = REFERENCE_SCORING_SEMANTICS_FAMILY_LATE_INTERACTION
        self.required_artifact_families = [
            REFERENCE_SCORING_REQUIRED_ARTIFACT_LATE_INTERACTION
        ]
        self.score_kind = REFERENCE_SCORING_SCORE_KIND_EXACT

    def __init__(out self, var kind: String) raises:
        self.kind = kind^
        self.family = reference_scoring_family_for_kind(self.kind)
        self.required_artifact_families = (
            required_artifact_families_for_reference_scoring_kind(self.kind)
        )
        self.score_kind = score_kind_for_reference_scoring_kind(self.kind)

    def requires_artifact_family(self, family: String) -> Bool:
        for required_family in self.required_artifact_families:
            if required_family == family:
                return True

        return False


def exact_late_interaction_reference_scoring_semantics(
) -> ReferenceScoringSemantics:
    return ReferenceScoringSemantics()
