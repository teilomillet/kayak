"""Rank supplied candidates through the existing Choice computation."""

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, model_validator

from . import decisions
from .decisions import Contract, DecisionRequest, DecisionResult, Identifier, ModelInfo, Text
from .errors import InputError


class RankingRequest(Contract):
    """Explicit context, question, and candidate descriptions; no inferred prompt."""

    state: Text
    instructions: Text
    candidates: dict[Identifier, Text] = Field(min_length=1, max_length=decisions.MAX_CANDIDATES)

    @model_validator(mode="after")
    def valid_decision(self) -> Self:
        # Choice owns the input rules and aggregate limits for both entry points.
        self.as_decision()
        return self

    def as_decision(self) -> DecisionRequest:
        """Return an independent /v1 request, preserving candidate text and order."""
        return DecisionRequest.model_validate(
            {
                "state": self.state,
                "questions": {
                    "rank": {
                        "instructions": self.instructions,
                        "criteria": self.candidates,
                    }
                },
            }
        )


class RankedCandidate(Contract):
    """One supplied ID with its scaled cosine and full-candidate-set softmax share."""

    id: str
    score: float
    probability: float


class RankingResult(Contract):
    """All candidates, best first; ties retain the submitted order.

    Selecting a prefix does not renormalize probabilities. The first item is
    the best supplied candidate, even when every candidate is unsuitable.
    """

    model: ModelInfo
    ranked: list[RankedCandidate] = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    calibration: Literal["none"] = "none"

    @model_validator(mode="after")
    def valid_ranking(self) -> Self:
        ids = [candidate.id for candidate in self.ranked]
        scores = [candidate.score for candidate in self.ranked]
        if len(set(ids)) != len(ids) or any(not candidate_id for candidate_id in ids):
            raise ValueError("ranked candidate IDs must be nonempty and unique")
        if scores != sorted(scores, reverse=True):
            raise ValueError("ranked candidates must be in descending score order")
        # Reuse the canonical score/distribution validation; no second definition.
        decisions.ChoiceAnswer(
            choice=ids[0],
            scores=dict(zip(ids, scores, strict=True)),
            probabilities={candidate.id: candidate.probability for candidate in self.ranked},
        )
        return self


def request_from(state: str, instructions: str, candidates: Mapping[str, str]) -> DecisionRequest:
    """Validate and snapshot a ranking before any model or network effects."""
    if not isinstance(candidates, Mapping):
        raise InputError("candidates must be a mapping of IDs to explicit text descriptions")
    return decisions.request_from(
        state, {"rank": {"instructions": instructions, "criteria": dict(candidates)}}
    )


def result_from(result: DecisionResult) -> RankingResult:
    """Project one checked Choice result into a stable ranking, without new inference."""
    answer = result.answers["rank"]
    ordered_ids = sorted(answer.scores, key=answer.scores.__getitem__, reverse=True)
    return RankingResult(
        model=result.model,
        input_tokens=result.input_tokens,
        calibration=result.calibration,
        ranked=[
            RankedCandidate(
                id=candidate_id,
                score=answer.scores[candidate_id],
                probability=answer.probabilities[candidate_id],
            )
            for candidate_id in ordered_ids
        ],
    )
