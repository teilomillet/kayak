"""Names the reference scoring semantics for Python search plans."""

from __future__ import annotations

from dataclasses import dataclass, field


REFERENCE_SCORING_SEMANTICS_FAMILY_LATE_INTERACTION = "late_interaction"
REFERENCE_SCORING_REQUIRED_ARTIFACT_LATE_INTERACTION = "late_interaction"
REFERENCE_SCORING_SCORE_KIND_EXACT = "exact_score"


@dataclass(frozen=True, slots=True)
class ReferenceScoringSemantics:
    kind: str
    family: str = field(init=False)
    required_artifact_families: tuple[str, ...] = field(init=False)
    score_kind: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind != "exact_late_interaction":
            raise ValueError(
                f"unsupported reference scoring semantics: {self.kind}"
            )

        object.__setattr__(
            self, "family", REFERENCE_SCORING_SEMANTICS_FAMILY_LATE_INTERACTION
        )
        object.__setattr__(
            self,
            "required_artifact_families",
            (REFERENCE_SCORING_REQUIRED_ARTIFACT_LATE_INTERACTION,),
        )
        object.__setattr__(self, "score_kind", REFERENCE_SCORING_SCORE_KIND_EXACT)

    def requires_artifact_family(self, family: str) -> bool:
        return family in self.required_artifact_families


def exact_late_interaction_reference_scoring_semantics(
) -> ReferenceScoringSemantics:
    return ReferenceScoringSemantics("exact_late_interaction")
