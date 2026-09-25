"""Numeric boundaries and ownership of validated decision results."""

import json
import math
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import ValidationError

from kayak import ChoiceAnswer, DecisionResult
from kayak.decisions import answer_from_scores


@pytest.mark.parametrize("field", ["scores", "probabilities"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.5", None])
def test_invalid_numbers_are_rejected_in_python_and_json(field: str, value: object) -> None:
    payload = {
        "choice": "first",
        "scores": {"first": 0.0, "second": 0.0},
        "probabilities": {"first": 0.5, "second": 0.5},
        field: {"first": value, "second": 0.5},
    }
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate(payload)
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "probability", [math.nextafter(0.0, -math.inf), math.nextafter(1.0, math.inf)]
)
def test_probability_bounds_are_exact_even_inside_distribution_tolerance(
    probability: float,
) -> None:
    # Both sums and softmax differences fall inside 1e-6; the range still must fail.
    probabilities = {"first": probability, "second": 1.0 - probability}
    scores = (
        {"first": -1000.0, "second": 0.0} if probability < 0 else {"first": 0.0, "second": -1000.0}
    )
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            choice=max(scores, key=scores.__getitem__), scores=scores, probabilities=probabilities
        )


@given(scores=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=256))
@example(scores=[0.0, 0.0])
@example(scores=[-1.7976931348623157e308, 1.7976931348623157e308])
@example(scores=[-0.0, 0.0, 5e-324])
def test_finite_score_distributions_preserve_values_order_and_roundtrip(
    scores: list[float],
) -> None:
    keys = [f"candidate-{i}" for i in range(len(scores))]
    answer = answer_from_scores(keys, scores)
    assert list(answer.scores) == list(answer.probabilities) == keys
    assert list(answer.scores.values()) == scores
    assert answer.choice == keys[scores.index(max(scores))]
    assert all(0 <= probability <= 1 for probability in answer.probabilities.values())
    assert math.isclose(math.fsum(answer.probabilities.values()), 1.0, abs_tol=1e-6)
    assert ChoiceAnswer.model_validate_json(answer.model_dump_json()) == answer
    scores.clear()
    assert len(answer.scores) == len(keys)


def test_probability_order_does_not_change_candidate_meaning() -> None:
    # Existing validation matches probabilities by ID, regardless of their wire order.
    answer = ChoiceAnswer(
        choice="first",
        scores={"first": 0.0, "second": 0.0},
        probabilities={"second": 0.5, "first": 0.5},
    )
    assert list(answer.probabilities) == ["second", "first"]
    assert ChoiceAnswer.model_validate_json(answer.model_dump_json()) == answer


@pytest.mark.parametrize("field", ["scores", "probabilities"])
def test_nested_results_copy_values_and_reject_later_mutations(field: str) -> None:
    result = DecisionResult.model_validate_json(
        (Path(__file__).parent / "fixtures/api_v1/response.json").read_bytes()
    )
    copied = DecisionResult.model_validate(result)
    answer = result.answers["department"]
    values = answer.scores if field == "scores" else answer.probabilities
    values["billing"] = float("nan")
    assert copied.answers["department"].scores == {"billing": 0.0, "technical": 0.0}
    assert copied.answers["department"].probabilities == {"billing": 0.5, "technical": 0.5}
    with pytest.raises(ValidationError):
        DecisionResult.model_validate(result)
