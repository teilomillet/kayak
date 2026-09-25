"""Typed CLM judgments compiled to the existing Choice computation and wire contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from . import decisions
from .decisions import (
    Choice,
    ChoiceAnswer,
    Contract,
    DecisionRequest,
    DecisionResult,
    Identifier,
    ModelInfo,
    Text,
)
from .errors import InputError


class Noul(Contract):
    """Compare false/true descriptions; the returned share is not calibrated truth."""

    type: Literal["noul"] = "noul"
    instructions: Text
    criteria: dict[Literal["false", "true"], str] | None = None

    @model_validator(mode="after")
    def valid_choice(self) -> Self:
        self._as_choice()
        return self

    def _as_choice(self) -> Choice:
        """Use the pinned CLM prefixes and false-then-true order, including defaults."""
        descriptions = self.criteria or {}
        candidates = {}
        for key in ("false", "true"):
            description = descriptions.get(key)
            if not description:
                prefix = "Yes. This is true: " if key == "true" else "No. This is false: "
                description = prefix + self.instructions.strip()
            candidates[key] = f"{key}: {description}"
        return Choice(instructions=self.instructions, criteria=candidates)


class Score(Contract):
    """An ordered rubric; score is the expected zero-based, equally spaced level."""

    type: Literal["score"] = "score"
    instructions: Text
    criteria: list[Text] = Field(min_length=2, max_length=decisions.MAX_CANDIDATES)

    @model_validator(mode="after")
    def valid_choice(self) -> Self:
        self._as_choice()
        return self

    def _as_choice(self) -> Choice:
        """Preserve descriptions verbatim and identify levels by their zero-based indices."""
        return Choice(
            instructions=self.instructions,
            criteria={str(index): text for index, text in enumerate(self.criteria)},
        )


def _check_distribution(scores: dict[str, float], probabilities: dict[str, float]) -> None:
    # Choice remains the single owner of finite-score and softmax consistency rules.
    ChoiceAnswer(
        choice=max(scores, key=scores.__getitem__), scores=scores, probabilities=probabilities
    )


class NoulAnswer(Contract):
    """The true candidate's softmax share, with both scores and shares retained.

    No boolean decision or abstention threshold is implied. A share of 0.5 is a tie.
    """

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)
    scores: dict[str, float] = Field(min_length=2, max_length=2)
    probabilities: dict[str, float] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid_distribution(self) -> Self:
        if list(self.scores) != ["false", "true"]:
            raise ValueError("Noul scores require false then true")
        _check_distribution(self.scores, self.probabilities)
        if self.noul != self.probabilities["true"]:
            raise ValueError("noul must equal the true candidate's probability")
        return self


class ScoreAnswer(Contract):
    """Expected rubric index and the full distribution, including possible ambiguity.

    A middle score can reflect disagreement between extremes, not a middle-level judgment.
    """

    type: Literal["score"] = "score"
    score: float = Field(ge=0, allow_inf_nan=False)
    legend: dict[str, Text] = Field(min_length=2, max_length=decisions.MAX_CANDIDATES)
    scores: dict[str, float] = Field(min_length=2, max_length=decisions.MAX_CANDIDATES)
    probabilities: dict[str, float] = Field(min_length=2, max_length=decisions.MAX_CANDIDATES)

    @model_validator(mode="after")
    def valid_distribution(self) -> Self:
        keys = [str(index) for index in range(len(self.legend))]
        if list(self.legend) != keys or list(self.scores) != keys:
            raise ValueError("Score legend and scores require consecutive levels starting at zero")
        _check_distribution(self.scores, self.probabilities)
        expected = sum(index * self.probabilities[key] for index, key in enumerate(keys))
        if not math.isclose(self.score, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("score must equal the probability-weighted rubric index")
        return self


JudgmentQuestion = Annotated[Choice | Noul | Score, Field(discriminator="type")]
JudgmentAnswer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]
_question: TypeAdapter[JudgmentQuestion] = TypeAdapter(JudgmentQuestion)


class JudgmentResult(Contract):
    """Typed Python answers; model identity and token use come from the Choice execution."""

    model: ModelInfo
    answers: dict[Identifier, JudgmentAnswer] = Field(
        min_length=1, max_length=decisions.MAX_QUESTIONS
    )
    input_tokens: int = Field(ge=0)
    calibration: Literal["none"] = "none"

    @model_validator(mode="after")
    def valid_ids(self) -> Self:
        if any(question_id.isspace() for question_id in self.answers):
            raise ValueError("answer IDs must not be blank")
        return self


class JudgmentRequest(Contract):
    """Validate named typed questions and their compiled limits without loading a model."""

    state: Text
    questions: dict[Identifier, JudgmentQuestion] = Field(
        min_length=1, max_length=decisions.MAX_QUESTIONS
    )

    @model_validator(mode="after")
    def valid_decision(self) -> Self:
        self.as_decision()
        return self

    def as_decision(self) -> DecisionRequest:
        """Return an independent Choice request using the pinned text-only CLM recipes."""
        # Frozen contracts contain mutable containers; recheck before compiling them.
        questions = {
            name: _question.validate_python(value) for name, value in self.questions.items()
        }
        return DecisionRequest(
            state=self.state,
            questions={
                name: question if isinstance(question, Choice) else question._as_choice()
                for name, question in questions.items()
            },
        )

    def decode(self, result: DecisionResult) -> JudgmentResult:
        """Validate an associated Choice result, then decode without further inference.

        Keep this request unchanged between compilation and decoding. The transport
        does not echo descriptions, so it cannot detect same-ID wording changes.
        """
        request = JudgmentRequest.model_validate(self)
        decision = request.as_decision()
        result = DecisionResult.model_validate(result)
        decisions.check_result(decision, result)
        answers: dict[str, JudgmentAnswer] = {}
        for name, question in request.questions.items():
            answer = result.answers[name]
            if isinstance(question, Noul):
                answers[name] = NoulAnswer(
                    noul=answer.probabilities["true"],
                    scores=answer.scores,
                    probabilities=answer.probabilities,
                )
            elif isinstance(question, Score):
                keys = list(answer.scores)
                answers[name] = ScoreAnswer(
                    score=sum(index * answer.probabilities[key] for index, key in enumerate(keys)),
                    legend=dict(zip(keys, question.criteria, strict=True)),
                    scores=answer.scores,
                    probabilities=answer.probabilities,
                )
            else:
                answers[name] = answer
        return JudgmentResult(
            model=result.model,
            input_tokens=result.input_tokens,
            calibration=result.calibration,
            answers=answers,
        )


def request_from(
    state: str, questions: Mapping[str, JudgmentQuestion | Mapping[str, object]]
) -> JudgmentRequest:
    """Snapshot typed input before effects and translate validation into InputError."""
    if not isinstance(questions, Mapping):
        raise InputError("questions must map IDs to Choice, Noul, or Score questions")
    try:
        return JudgmentRequest.model_validate({"state": state, "questions": dict(questions)})
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        message = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'request'}: "
            + (
                "question type must be 'choice', 'noul', or 'score'"
                if error["type"] == "union_tag_invalid"
                else error["msg"]
            )
            for error in errors
        )
        raise InputError(message) from None
