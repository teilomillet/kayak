"""Agreement with frozen outputs from independent numerical libraries.

Regenerate with benchmarks/eval_references.py. These tests require no optional
reference libraries, model weights, training data, or network access.
"""

import json
from pathlib import Path

import pytest

from kayak.eval import MetricInput, MetricSample, default_metrics, score_metrics
from kayak.eval._benchmark import accuracy_interval, mcnemar_exact
from kayak.eval._metrics import classification_summary
from kayak.eval._ranking import assess_ranking

REFERENCE = json.loads((Path(__file__).parent / "fixtures/eval-reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["classification"])
def test_classification_matches_sklearn(case: dict[str, object]) -> None:
    labels, gold, choices, expected = (
        case["labels"],
        case["gold"],
        case["choices"],
        case["expected"],
    )
    assert isinstance(labels, list) and isinstance(gold, list) and isinstance(choices, list)
    assert isinstance(expected, dict)
    actual = classification_summary(tuple(labels), tuple(gold), tuple(choices))
    for metric, value in expected.items():
        assert actual[metric] == pytest.approx(value, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("case", REFERENCE["probabilities"])
def test_probability_metrics_match_sklearn(case: dict[str, object]) -> None:
    labels, gold, vectors, expected = (
        case["labels"],
        case["gold"],
        case["probabilities"],
        case["expected"],
    )
    assert isinstance(labels, list) and isinstance(gold, list) and isinstance(vectors, list)
    assert isinstance(expected, dict)
    data = MetricInput(
        tuple(labels),
        tuple(
            MetricSample(str(index), label, labels[0], tuple(vector), None)
            for index, (label, vector) in enumerate(zip(gold, vectors, strict=True))
        ),
    )
    actual = score_metrics(data, default_metrics())
    for metric, value in expected.items():
        assert actual[metric]["value"] == pytest.approx(value, abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("case", REFERENCE["ranking"])
def test_linear_gain_ndcg_matches_sklearn(case: dict[str, object]) -> None:
    order, relevance, k = case["order"], case["relevance"], case["k"]
    assert isinstance(order, list) and isinstance(relevance, dict) and isinstance(k, int)
    actual = assess_ranking(order, relevance, k=k)
    assert actual.metrics["ndcg_at_k"] == pytest.approx(case["ndcg_at_k"], abs=1e-12, rel=1e-12)


@pytest.mark.parametrize("case", REFERENCE["mcnemar"])
def test_exact_mcnemar_matches_scipy(case: dict[str, float]) -> None:
    assert mcnemar_exact(int(case["fixed"]), int(case["regressed"])) == pytest.approx(
        case["p"], abs=0, rel=1e-12
    )


@pytest.mark.parametrize("case", REFERENCE["wilson"])
def test_wilson_interval_matches_scipy(case: dict[str, float]) -> None:
    actual = accuracy_interval(int(case["correct"]), int(case["total"]))
    for endpoint in ("lower", "upper"):
        assert actual[endpoint] == pytest.approx(case[endpoint], abs=1e-12, rel=1e-12)
