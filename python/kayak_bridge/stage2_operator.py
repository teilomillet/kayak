"""Compatibility view over the legacy combined Python stage-2 operator."""

from __future__ import annotations

from dataclasses import dataclass, field

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


STAGE2_OPERATOR_FAMILY_IDENTITY = STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY
STAGE2_OPERATOR_FAMILY_HYBRID = "hybrid"
STAGE2_OPERATOR_FAMILY_LATE_INTERACTION = (
    STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION
)
STAGE2_OPERATOR_FAMILY_TEXT = STAGE3_VERIFIER_FAMILY_TEXT

STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT = STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT
STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION = (
    STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION
)


def reference_scoring_semantics_for_stage2_operator_kind(
    kind: str,
) -> ReferenceScoringSemantics:
    _ = Stage2Operator(kind)
    return exact_late_interaction_reference_scoring_semantics()


def stage2_reference_operator_for_stage2_operator_kind(
    kind: str,
) -> Stage2ReferenceOperator:
    if kind in {"noop_topk", "clause_text"}:
        return noop_topk_stage2_reference_operator()
    if kind in {
        "exact_late_interaction",
        "exact_late_interaction_clause_text",
    }:
        return exact_late_interaction_stage2_reference_operator()
    raise ValueError(f"unsupported stage-2 operator: {kind}")


def stage3_verifier_for_stage2_operator_kind(kind: str) -> Stage3VerifierOperator:
    if kind in {"noop_topk", "exact_late_interaction"}:
        return none_stage3_verifier_operator()
    if kind in {"clause_text", "exact_late_interaction_clause_text"}:
        return clause_text_stage3_verifier_operator()
    raise ValueError(f"unsupported stage-2 operator: {kind}")


def combined_stage2_operator_kind(
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) -> str:
    if stage2_reference_operator.kind == "noop_topk":
        if stage3_verifier.kind == "none":
            return "noop_topk"
        if stage3_verifier.kind == "clause_text":
            return "clause_text"
    if stage2_reference_operator.kind == "exact_late_interaction":
        if stage3_verifier.kind == "none":
            return "exact_late_interaction"
        if stage3_verifier.kind == "clause_text":
            return "exact_late_interaction_clause_text"

    raise ValueError(
        "unsupported compatibility stage-2 composition: "
        f"{stage2_reference_operator.kind} + {stage3_verifier.kind}"
    )


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


def stage2_operator_for_components(
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) -> Stage2Operator:
    return Stage2Operator(
        combined_stage2_operator_kind(stage2_reference_operator, stage3_verifier)
    )
