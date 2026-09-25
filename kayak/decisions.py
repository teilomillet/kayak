"""Decision inputs, outputs, and their pure validation and scoring functions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import InferenceError, InputError

MAX_QUESTIONS = 32
MAX_CANDIDATES = 256  # Total across all questions in one request.
MAX_REQUEST_CHARS = 262_144
MAX_TOTAL_TOKENS = 16_384
MAX_REQUEST_BYTES = 1_048_576  # Serialized request limit shared by HTTP and CLI.
Text = Annotated[str, Field(min_length=1, max_length=65_536)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )


class Choice(Contract):
    """Choose an option ID using its explicit description and the instructions."""

    type: Literal["choice"] = "choice"
    instructions: Text
    criteria: dict[Identifier, Text] = Field(min_length=1, max_length=MAX_CANDIDATES)

    @model_validator(mode="after")
    def nonblank(self) -> Self:
        # Field validation already excludes empty strings. Avoid copying large
        # text values merely to ask whether they contain a non-whitespace character.
        if self.instructions.isspace() or any(
            candidate_id.isspace() or description.isspace()
            for candidate_id, description in self.criteria.items()
        ):
            raise ValueError("instructions, option IDs, and descriptions must not be blank")
        return self


class DecisionRequest(Contract):
    """A validated request value for Python, JSON files, and HTTP; no model is loaded."""

    state: Text
    questions: dict[Identifier, Choice] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @model_validator(mode="after")
    def limits(self) -> Self:
        if self.state.isspace() or any(question_id.isspace() for question_id in self.questions):
            raise ValueError("state and question IDs must not be blank")
        candidate_count = sum(len(question.criteria) for question in self.questions.values())
        if candidate_count > MAX_CANDIDATES:
            raise ValueError(f"at most {MAX_CANDIDATES} candidates are allowed per request")
        total_characters = len(self.state)
        for question_id, question in self.questions.items():
            total_characters += len(question_id) + len(question.instructions)
            total_characters += sum(
                len(candidate_id) + len(description)
                for candidate_id, description in question.criteria.items()
            )
        if total_characters > MAX_REQUEST_CHARS:
            raise ValueError(f"request exceeds {MAX_REQUEST_CHARS} characters")
        return self


class ModelInfo(Contract):
    """Artifact identity, input recipe, and the model's resolved device and precision."""

    id: str
    revision: str
    fingerprint: str
    encoder: str
    encoder_revision: str
    input_recipe: Literal["clm-choice-v1"] = "clm-choice-v1"
    device: str
    dtype: str


class ChoiceAnswer(Contract):
    """The first highest-scoring ID, all scores, and uncalibrated candidate-set shares."""

    type: Literal["choice"] = "choice"
    choice: str
    scores: dict[str, float] = Field(min_length=1)
    probabilities: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_distribution(self) -> Self:
        if self.scores.keys() != self.probabilities.keys():
            raise ValueError("score and probability IDs differ")
        if not all(map(math.isfinite, self.scores.values())):
            raise ValueError("scores must be finite")
        probabilities = self.probabilities.values()
        if (
            not all(map(math.isfinite, probabilities))
            or min(probabilities) < 0
            or max(probabilities) > 1
        ):
            raise ValueError("probabilities must be finite and within [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        if self.choice != max(self.scores, key=self.scores.__getitem__):
            raise ValueError("choice must be the first highest-scoring option")
        # Probabilities have one defined meaning in this first native contract.
        expected_probabilities = softmax(list(self.scores.values()))
        if any(
            not math.isclose(self.probabilities[candidate_id], probability, abs_tol=1e-6)
            for candidate_id, probability in zip(self.scores, expected_probabilities, strict=True)
        ):
            raise ValueError("probabilities must be the softmax of the scores")
        return self


class DecisionResult(Contract):
    """Answers keyed by the caller's question IDs, with model identity and token use."""

    model: ModelInfo
    answers: dict[str, ChoiceAnswer] = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    calibration: Literal["none"] = "none"

    @field_validator("answers")
    @classmethod
    def valid_ids(cls, value: dict[str, ChoiceAnswer]) -> dict[str, ChoiceAnswer]:
        if any(not question_id for question_id in value):
            raise ValueError("answer IDs must not be empty")
        return value


def validation_message(exc: ValidationError) -> str:
    """Describe invalid fields without echoing request text or tensor values."""
    messages: list[str] = []
    for error in exc.errors(include_input=False, include_url=False):
        location = error["loc"]
        message = error["msg"]
        if location == ("state",) and error["type"] == "string_type":
            message = "state must be text; serialize structured state explicitly with json.dumps"
        elif location[-1:] == ("type",) and error["type"] == "literal_error":
            message = (
                "decide supports only Choice questions (type='choice'); "
                "use the Python judge API for Noul/Score"
            )
        elif (
            len(location) == 4
            and location[0] == "questions"
            and location[2] == "criteria"
            and error["type"] == "string_type"
        ):
            message = "each candidate needs an explicit text description"
        messages.append(f"{'.'.join(map(str, location)) or 'request'}: {message}")
    return "; ".join(messages)


def request_from(
    state: str, questions: Mapping[str, Choice | Mapping[str, object]]
) -> DecisionRequest:
    """Validate and snapshot caller-owned questions; translate failures to InputError."""
    # Contract.revalidate_instances copies and rechecks typed inputs too.
    # The validated request owns its dictionaries, even if the caller mutates theirs.
    if not isinstance(questions, Mapping):
        raise InputError("questions must be a mapping of IDs to Choice questions")
    payload = {"state": state, "questions": dict(questions)}
    try:
        return DecisionRequest.model_validate(payload)
    except ValidationError as exc:
        raise InputError(validation_message(exc)) from exc


def softmax(scores: list[float]) -> list[float]:
    """Compute stable softmax shares in input order, rejecting empty/nonfinite scores."""
    if not scores or not all(map(math.isfinite, scores)):
        raise InferenceError("model scores must be a nonempty finite sequence")
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def answer_from_scores(keys: list[str], scores: list[float]) -> ChoiceAnswer:
    """Pair candidate IDs with scores and softmax shares; the first maximum wins."""
    if len(keys) != len(scores) or len(set(keys)) != len(keys):
        raise InferenceError("model score count does not match candidate IDs")
    probabilities = softmax(scores)
    return ChoiceAnswer(
        choice=keys[max(range(len(scores)), key=scores.__getitem__)],
        scores=dict(zip(keys, scores, strict=True)),
        probabilities=dict(zip(keys, probabilities, strict=True)),
    )


def check_result(request: DecisionRequest, result: DecisionResult) -> None:
    """Require matching question IDs and unchanged candidate IDs/order in a response."""
    if set(request.questions) != set(result.answers):
        raise InferenceError("answer IDs do not match the request")
    for question_id, question in request.questions.items():
        if list(question.criteria) != list(result.answers[question_id].scores):
            raise InferenceError("candidate IDs or their order do not match the request")
