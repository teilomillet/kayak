"""Classification metrics checked against arithmetic and a covariance reference."""

import math
from pathlib import Path
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_eval_runner import MODEL

from kayak import Choice, DecisionResult
from kayak.decisions import answer_from_scores
from kayak.eval import Example, Report, Suite, compare, evaluate, load_report, summarize
from kayak.eval._schema import Attempt, Failure, Observation, Protocol


def classification_report(
    gold: list[str], predicted: list[str | None], *, labels: tuple[str, ...] = ("a", "b", "c")
) -> Report:
    suite = Suite(
        name="metric-arithmetic",
        split="dev",
        question=Choice(instructions="Select intent", criteria={label: label for label in labels}),
        examples=[Example(id=str(i), text=str(i), label=label) for i, label in enumerate(gold)],
    )
    observations = []
    for i, label in enumerate(predicted):
        if label is None:
            continue
        result = DecisionResult(
            model=MODEL,
            answers={
                "intent": answer_from_scores(list(labels), [float(key == label) for key in labels])
            },
            input_tokens=1,
        )
        observations.append(Observation(id=str(i), attempts=[Attempt(seconds=1.0, result=result)]))
    return Report(
        created_at="fixture",
        suite=suite,
        suite_sha256=suite.sha256,
        protocol=Protocol(warmups=0),
        transport="custom",
        environment={},
        observations=observations,
    )


def test_imbalanced_multiclass_arithmetic_and_zero_support() -> None:
    # Matrix rows: A=[2,2,0], B=[0,1,1], C=[0,0,1]. Gold support is 4:2:1.
    report = classification_report(list("aaaabbc"), list("aabbbcc"))
    summary = summarize(report)
    assert summary["accuracy"] == pytest.approx(4 / 7)
    assert summary["balanced_accuracy"] == pytest.approx(2 / 3)
    assert summary["macro_f1"] == pytest.approx(26 / 45)
    assert summary["weighted_f1"] == pytest.approx(62 / 105)
    assert summary["matthews_correlation"] == pytest.approx(12 / math.sqrt(896))
    assert summary["zero_recall_labels"] == []
    assert summary["predicted_label_count"] == 3
    assert summary["largest_prediction_share"] == pytest.approx(3 / 7)
    assert summary["majority_class_accuracy"] == pytest.approx(4 / 7)
    assert summary["uniform_chance_accuracy"] == pytest.approx(1 / 3)

    expanded = summarize(
        classification_report(list("aaaabbc"), list("aabbbcc"), labels=("a", "b", "c", "d"))
    )
    assert expanded["macro_f1"] == pytest.approx(13 / 30)
    assert expanded["uniform_chance_accuracy"] == 0.25
    for key in ("balanced_accuracy", "weighted_f1", "matthews_correlation", "zero_recall_labels"):
        assert expanded[key] == summary[key]


@pytest.mark.parametrize(
    ("gold", "predicted", "accuracy", "balanced", "weighted", "mcc", "zero_recall"),
    [
        ("aaab", "aaab", 1.0, 1.0, 1.0, 1.0, []),
        ("aaab", "aaaa", 0.75, 0.5, 9 / 14, 0.0, ["b"]),
        ("aabb", "bbaa", 0.0, 0.0, 0.0, -1.0, ["a", "b"]),
        ("aaaa", "aaaa", 1.0, 1.0, 1.0, 0.0, []),
        ("aaaa", "bbbb", 0.0, 0.0, 0.0, 0.0, ["a"]),
    ],
)
def test_perfect_constant_and_inverted_predictions(
    gold: str,
    predicted: str,
    accuracy: float,
    balanced: float,
    weighted: float,
    mcc: float,
    zero_recall: list[str],
) -> None:
    summary = summarize(classification_report(list(gold), list(predicted)))
    assert summary["accuracy"] == accuracy
    assert summary["balanced_accuracy"] == balanced
    assert summary["weighted_f1"] == pytest.approx(weighted)
    assert summary["matthews_correlation"] == mcc
    assert summary["zero_recall_labels"] == zero_recall


def test_failures_missing_rows_and_later_success_keep_the_first_attempt_denominator() -> None:
    report = classification_report(list("aabb"), ["a", None, "b", "b"], labels=("a", "b"))
    # One missing row and one failed first attempt, followed by a successful retry.
    report.observations[-1].attempts.insert(0, Attempt(seconds=1.0, error=Failure(type="Error")))
    summary = summarize(report)
    assert summary["failed_or_missing_examples"] == 2
    assert summary["accuracy"] == summary["balanced_accuracy"] == 0.5
    assert summary["weighted_f1"] == pytest.approx(2 / 3)
    assert summary["matthews_correlation"] == pytest.approx(4 / math.sqrt(80))
    assert summary["largest_prediction_share"] == 0.25
    assert summary["predicted_label_count"] == 2
    report.observations = []
    empty = summarize(report)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "weighted_f1",
        "matthews_correlation",
        "predicted_label_count",
        "largest_prediction_share",
    ):
        assert empty[key] == 0
    assert empty["zero_recall_labels"] == ["a", "b"]


@given(
    st.lists(
        st.tuples(st.sampled_from("abc"), st.sampled_from(["a", "b", "c", None])),
        min_size=1,
        max_size=40,
    )
)
def test_mcc_matches_centered_one_hot_vectors(pairs: list[tuple[str, str | None]]) -> None:
    # Independent definition: normalized dot product of centered indicator vectors.
    # It does not reuse the confusion-matrix formula used by the implementation.
    gold = [left for left, _ in pairs]
    predicted = [right for _, right in pairs]
    left_vector: list[float] = []
    right_vector: list[float] = []
    for label in ("a", "b", "c", None):
        left_mean = gold.count(label) / len(gold) if label is not None else 0.0
        right_mean = predicted.count(label) / len(predicted)
        left_vector.extend(float(value == label) - left_mean for value in gold)
        right_vector.extend(float(value == label) - right_mean for value in predicted)
    numerator = sum(left * right for left, right in zip(left_vector, right_vector, strict=True))
    denominator = math.sqrt(sum(x * x for x in left_vector) * sum(x * x for x in right_vector))
    expected = numerator / denominator if denominator else 0.0
    actual = summarize(classification_report(gold, predicted))["matthews_correlation"]
    assert actual == pytest.approx(expected, abs=1e-12)


def test_frozen_schema_two_artifacts_can_be_resummarized_without_writes() -> None:
    path = Path(__file__).resolve().parents[1] / "benchmarks/reports/2026-09-25-mock/complete"
    original = (path / "report.json").read_bytes()
    report = load_report(path)
    assert report.schema_version == 2
    assert "weighted_f1" not in report.summary
    assert summarize(report)["weighted_f1"] == pytest.approx(1 / 3)
    comparison = compare(path, path)
    assert comparison["paired_outcomes"] == {
        "both_correct": 2,
        "fixed": 0,
        "regressed": 0,
        "both_wrong": 2,
    }
    assert comparison["mean_latency_speedup"] is None
    assert (path / "report.json").read_bytes() == original


def test_comparison_counts_paired_fixes_and_regressions_once(tmp_path: Path) -> None:
    # One fixed, one regressed, one unchanged correct, one unchanged wrong.
    for name, predictions, seed in (("before", "aaba", 1), ("after", "bbba", 2)):
        fixture = classification_report(list("abbb"), list(predictions))
        results = {row.id: row.attempts[0].result for row in fixture.observations}
        backend = Mock()
        backend.decide.side_effect = lambda *, state, questions, results=results: results[state]
        evaluate(backend, fixture.suite, output=tmp_path / name, warmups=0, repeats=3, seed=seed)
    comparison = compare(tmp_path / "before", tmp_path / "after")
    assert comparison["paired_outcomes"] == {
        "both_correct": 1,
        "fixed": 1,
        "regressed": 1,
        "both_wrong": 1,
    }
    deltas = comparison["quality_delta"]
    assert isinstance(deltas, dict)
    assert deltas["accuracy"] == 0.0
    # Identical accuracy can hide a gain on the majority intent and loss of the minority.
    assert deltas["balanced_accuracy"] == pytest.approx(-1 / 3)
