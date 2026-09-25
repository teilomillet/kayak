"""Provider-reported judgments, separate from the native CLM score contract."""

import math
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from ..decisions import Contract
from ..errors import InferenceError
from ..judgments import JudgmentRequest

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ProviderChoice(Contract):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability] | None = None


class ProviderNoul(Contract):
    type: Literal["noul"] = "noul"
    noul: Probability
    probabilities: dict[str, Probability] | None = None


class ProviderScore(Contract):
    type: Literal["score"] = "score"
    score: float = Field(ge=0, allow_inf_nan=False)
    legend: dict[str, str]
    probabilities: dict[str, Probability] | None = None


ProviderAnswer = Annotated[
    ProviderChoice | ProviderNoul | ProviderScore, Field(discriminator="type")
]


class ProviderResult(Contract):
    """Validated answer fields plus a detached copy of the provider's JSON body.

    Model and usage are provider reports, not independently verified identity or
    accounting. Calibration is unknown to Kayak. No native CLM scores are invented.
    """

    provider: Literal["laya", "jev"]
    model: str | None = None
    answers: dict[str, ProviderAnswer]
    input_tokens: int | None = Field(default=None, ge=0)
    calibration: Literal["unknown"] = "unknown"
    raw: dict[str, JsonValue]


_body = TypeAdapter(dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False))
_answer: TypeAdapter[ProviderAnswer] = TypeAdapter(ProviderAnswer)


def _check_probabilities(answer: ProviderAnswer, keys: list[str], rounding: float | None) -> None:
    """Check supplied evidence, using the provider's precision only when known."""
    probabilities = answer.probabilities
    if probabilities is None:
        return
    if set(probabilities) != set(keys):
        raise ValueError("probability IDs differ")
    if isinstance(answer, ProviderChoice):
        if probabilities[answer.choice] != max(probabilities.values()):
            raise ValueError("choice is not a maximum")
    # Laya's source rounds each value independently to 4dp. Jev specifies
    # no precision, so its cross-field arithmetic remains unverified.
    if rounding is None:
        return
    if abs(sum(probabilities.values()) - 1) > len(keys) * rounding + 1e-6:
        raise ValueError("probability mass differs")
    if isinstance(answer, ProviderScore):
        expected = sum(index * probabilities[key] for index, key in enumerate(keys))
        tolerance = rounding * (1 + sum(range(len(keys)))) + 1e-6
        if abs(answer.score - expected) > tolerance:
            raise ValueError("score differs from distribution")
    elif isinstance(answer, ProviderNoul) and not math.isclose(
        answer.noul, probabilities["true"], abs_tol=rounding + 1e-6
    ):
        raise ValueError("noul differs from distribution")


def _answers(
    request: JudgmentRequest, body: dict[str, JsonValue], rounding: float | None
) -> dict[str, ProviderAnswer]:
    returned = body.get("answers")
    if not isinstance(returned, dict) or returned.keys() != request.questions.keys():
        raise ValueError("answer IDs differ")
    answers: dict[str, ProviderAnswer] = {}
    for name, question in request.questions.items():
        raw = returned[name]
        if not isinstance(raw, dict) or raw.get("type") != question.type:
            raise ValueError("answer type differs")
        # Provider-only fields stay in raw; no generic confidence statistic is implied.
        fields = {"type", "probabilities"}
        fields.update({"choice"} if question.type == "choice" else {question.type})
        if question.type == "score":
            fields.add("legend")
        answer = _answer.validate_python(
            {key: value for key, value in raw.items() if key in fields}
        )
        if question.type == "choice":
            assert isinstance(answer, ProviderChoice)
            keys = list(question.criteria)
            if answer.choice not in keys:
                raise ValueError("unknown choice")
        elif question.type == "score":
            assert isinstance(answer, ProviderScore)
            keys = [str(index) for index in range(len(question.criteria))]
            if answer.legend != dict(zip(keys, question.criteria, strict=True)):
                raise ValueError("rubric differs")
            if answer.score > len(keys) - 1:
                raise ValueError("score exceeds rubric")
        else:
            keys = ["false", "true"]
        _check_probabilities(answer, keys, rounding)
        answers[name] = answer
    return answers


def decode(
    provider: Literal["laya", "jev"], request: JudgmentRequest, response: object
) -> ProviderResult:
    """Check IDs, types, rubric and reported evidence without rewriting provider values."""
    try:
        body = (
            _body.validate_json(response, strict=True)
            if isinstance(response, bytes)
            else _body.validate_python(response, strict=True)
        )
        usage = body.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ValueError("invalid usage")
        return ProviderResult.model_validate(
            {
                "provider": provider,
                "model": body.get("model"),
                "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
                "answers": _answers(request, body, 0.00005 if provider == "laya" else None),
                "raw": body,
            }
        )
    except (ValidationError, ValueError, TypeError):
        # Do not include customer input or arbitrary provider content in validation errors.
        raise InferenceError(f"Invalid {provider} judgment response") from None
