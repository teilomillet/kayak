# Stage-3 verifiers rerank a bounded window after reference scoring.

from std.collections import List


comptime STAGE3_VERIFIER_FAMILY_IDENTITY = "identity"
comptime STAGE3_VERIFIER_FAMILY_TEXT = "text"
comptime STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT = "document_text"


def stage3_verifier_family_for_kind(kind: String) raises -> String:
    if kind == "none":
        return STAGE3_VERIFIER_FAMILY_IDENTITY
    if kind == "clause_text":
        return STAGE3_VERIFIER_FAMILY_TEXT

    raise Error("unknown stage-3 verifier kind: " + kind)


def required_artifact_families_for_stage3_verifier_kind(
    kind: String
) raises -> List[String]:
    if kind == "none":
        return List[String]()
    if kind == "clause_text":
        return [STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT]

    raise Error("unknown stage-3 verifier kind: " + kind)


def stage3_verifier_requires_query_text(kind: String) raises -> Bool:
    if kind == "clause_text":
        return True
    if kind == "none":
        return False

    raise Error("unknown stage-3 verifier kind: " + kind)


struct Stage3VerifierOperator(Copyable):
    var kind: String
    var family: String
    var required_artifact_families: List[String]
    var requires_query_text: Bool

    def __init__(out self):
        self.kind = "none"
        self.family = STAGE3_VERIFIER_FAMILY_IDENTITY
        self.required_artifact_families = List[String]()
        self.requires_query_text = False

    def __init__(out self, var kind: String) raises:
        self.kind = kind^
        self.family = stage3_verifier_family_for_kind(self.kind)
        self.required_artifact_families = (
            required_artifact_families_for_stage3_verifier_kind(self.kind)
        )
        self.requires_query_text = stage3_verifier_requires_query_text(self.kind)

    def requires_artifact_family(self, family: String) -> Bool:
        for required_family in self.required_artifact_families:
            if required_family == family:
                return True

        return False


def none_stage3_verifier_operator() -> Stage3VerifierOperator:
    return Stage3VerifierOperator()


def clause_text_stage3_verifier_operator() raises -> Stage3VerifierOperator:
    return Stage3VerifierOperator("clause_text")
