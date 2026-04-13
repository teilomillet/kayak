"""Owns explicit stage-2 refinement operators for Python search plans."""

from __future__ import annotations

from dataclasses import dataclass, field


STAGE2_OPERATOR_FAMILY_IDENTITY = "identity"
STAGE2_OPERATOR_FAMILY_HYBRID = "hybrid"
STAGE2_OPERATOR_FAMILY_LATE_INTERACTION = "late_interaction"
STAGE2_OPERATOR_FAMILY_TEXT = "text"

STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT = "document_text"
STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION = "late_interaction"


@dataclass(frozen=True, slots=True)
class Stage2Operator:
    kind: str
    family: str = field(init=False)
    required_artifact_families: tuple[str, ...] = field(init=False)
    requires_query_text: bool = field(init=False)
    is_exact_reference: bool = field(init=False)

    def __post_init__(self) -> None:
        family: str
        required_artifacts: tuple[str, ...]
        requires_query_text: bool
        is_exact_reference: bool

        if self.kind == "noop_topk":
            family = STAGE2_OPERATOR_FAMILY_IDENTITY
            required_artifacts = ()
            requires_query_text = False
            is_exact_reference = False
        elif self.kind == "exact_late_interaction_clause_text":
            family = STAGE2_OPERATOR_FAMILY_HYBRID
            required_artifacts = (
                STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION,
                STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
            )
            requires_query_text = True
            is_exact_reference = False
        elif self.kind == "exact_late_interaction":
            family = STAGE2_OPERATOR_FAMILY_LATE_INTERACTION
            required_artifacts = (STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION,)
            requires_query_text = False
            is_exact_reference = True
        elif self.kind == "clause_text":
            family = STAGE2_OPERATOR_FAMILY_TEXT
            required_artifacts = (STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT,)
            requires_query_text = True
            is_exact_reference = False
        else:
            raise ValueError(f"unsupported stage-2 operator: {self.kind}")

        object.__setattr__(self, "family", family)
        object.__setattr__(
            self, "required_artifact_families", required_artifacts
        )
        object.__setattr__(self, "requires_query_text", requires_query_text)
        object.__setattr__(self, "is_exact_reference", is_exact_reference)

    def requires_artifact_family(self, family: str) -> bool:
        return family in self.required_artifact_families


def noop_topk_stage2_operator() -> Stage2Operator:
    return Stage2Operator("noop_topk")


def exact_late_interaction_stage2_operator() -> Stage2Operator:
    return Stage2Operator("exact_late_interaction")


def exact_late_interaction_clause_text_stage2_operator() -> Stage2Operator:
    return Stage2Operator("exact_late_interaction_clause_text")


def clause_text_stage2_operator() -> Stage2Operator:
    return Stage2Operator("clause_text")
