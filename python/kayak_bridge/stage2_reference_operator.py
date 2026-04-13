"""Owns explicit stage-2 reference operators for Python search plans."""

from __future__ import annotations

from dataclasses import dataclass, field


STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY = "identity"
STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION = "late_interaction"
STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION = "late_interaction"


@dataclass(frozen=True, slots=True)
class Stage2ReferenceOperator:
    kind: str
    family: str = field(init=False)
    required_artifact_families: tuple[str, ...] = field(init=False)
    executes_reference_scoring: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.kind == "noop_topk":
            family = STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY
            required_artifacts = ()
            executes_reference_scoring = False
        elif self.kind == "exact_late_interaction":
            family = STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION
            required_artifacts = (
                STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
            )
            executes_reference_scoring = True
        else:
            raise ValueError(
                f"unsupported stage-2 reference operator: {self.kind}"
            )

        object.__setattr__(self, "family", family)
        object.__setattr__(
            self, "required_artifact_families", required_artifacts
        )
        object.__setattr__(
            self, "executes_reference_scoring", executes_reference_scoring
        )

    def requires_artifact_family(self, family: str) -> bool:
        return family in self.required_artifact_families


def noop_topk_stage2_reference_operator() -> Stage2ReferenceOperator:
    return Stage2ReferenceOperator("noop_topk")


def exact_late_interaction_stage2_reference_operator(
) -> Stage2ReferenceOperator:
    return Stage2ReferenceOperator("exact_late_interaction")
