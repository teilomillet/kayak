"""Owns explicit stage-3 verifier operators for Python search plans."""

from __future__ import annotations

from dataclasses import dataclass, field


STAGE3_VERIFIER_FAMILY_IDENTITY = "identity"
STAGE3_VERIFIER_FAMILY_TEXT = "text"
STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT = "document_text"


@dataclass(frozen=True, slots=True)
class Stage3VerifierOperator:
    kind: str
    family: str = field(init=False)
    required_artifact_families: tuple[str, ...] = field(init=False)
    requires_query_text: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.kind == "none":
            family = STAGE3_VERIFIER_FAMILY_IDENTITY
            required_artifacts = ()
            requires_query_text = False
        elif self.kind == "clause_text":
            family = STAGE3_VERIFIER_FAMILY_TEXT
            required_artifacts = (STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT,)
            requires_query_text = True
        else:
            raise ValueError(f"unsupported stage-3 verifier: {self.kind}")

        object.__setattr__(self, "family", family)
        object.__setattr__(
            self, "required_artifact_families", required_artifacts
        )
        object.__setattr__(self, "requires_query_text", requires_query_text)

    def requires_artifact_family(self, family: str) -> bool:
        return family in self.required_artifact_families


def none_stage3_verifier_operator() -> Stage3VerifierOperator:
    return Stage3VerifierOperator("none")


def clause_text_stage3_verifier_operator() -> Stage3VerifierOperator:
    return Stage3VerifierOperator("clause_text")
