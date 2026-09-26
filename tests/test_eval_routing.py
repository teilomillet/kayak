"""Independent binary references, failure accounting, score ties, and CLI boundaries."""

import json
from pathlib import Path

import pytest

from kayak import Choice
from kayak.eval import Example, Prediction, PredictionSet, Suite, _cli, routing_summary


def predictions(gold: list[str], choices: list[str | None]) -> PredictionSet:
    return PredictionSet(
        system="controlled",
        method="fixture",
        suite=Suite(
            name="routing-fixture",
            split="test",
            question=Choice(instructions="route", criteria={"a": "a", "b": "b", "oos": "reject"}),
            examples=[
                Example(id=str(i), text=f"input {i}", label=label) for i, label in enumerate(gold)
            ],
        ),
        predictions=[Prediction(id=str(i), choice=choice) for i, choice in enumerate(choices)],
    )


def test_every_routing_outcome_and_missing_answer() -> None:
    data = predictions(["a"] * 4 + ["oos"] * 3, ["a", "b", "oos", None, "oos", "a", None])
    report = routing_summary(data, reject_label="oos")
    assert report["counts"] == {
        "in_scope": 4,
        "out_of_scope": 3,
        "correct_routes": 1,
        "wrong_routes": 1,
        "false_rejections": 1,
        "correct_rejections": 1,
        "false_accepts": 1,
        "in_scope_failures": 1,
        "out_of_scope_failures": 1,
    }
    assert report["complete"] is False
    assert report["out_of_scope_f1"] == 2 / 5
    rates = report["rates"]
    assert isinstance(rates, dict)
    for name, value, numerator, denominator in [
        ("in_scope_accuracy", 1 / 4, 1, 4),
        ("out_of_scope_recall", 1 / 3, 1, 3),
        ("out_of_scope_precision", 1 / 2, 1, 2),
        ("routing_precision", 1 / 3, 1, 3),
        ("routing_coverage", 3 / 7, 3, 7),
        ("failure_rate", 2 / 7, 2, 7),
    ]:
        assert rates[name]["value"] == value
        assert rates[name]["numerator"] == numerator
        assert rates[name]["denominator"] == denominator
        assert rates[name]["interval_95"]["lower"] <= value <= rates[name]["interval_95"]["upper"]
    for name in ("out_of_scope_false_accept_rate", "in_scope_false_reject_rate"):
        assert rates[name]["value"] is None
        assert rates[name]["interval_95"] is None
    assert report["unobserved_gold_labels"] == ["b"]


@pytest.mark.parametrize(
    "gold,choices,missing_rate",
    [
        (["a"], ["a"], "out_of_scope_recall"),
        (["oos"], ["oos"], "in_scope_accuracy"),
        (["a", "oos"], ["oos", "oos"], "routing_precision"),
        (["a", "oos"], ["a", "a"], "out_of_scope_precision"),
    ],
)
def test_undefined_rates_are_null(gold: list[str], choices: list[str], missing_rate: str) -> None:
    report = routing_summary(predictions(gold, list(choices)), reject_label="oos")
    rates = report["rates"]
    assert isinstance(rates, dict)
    assert rates[missing_rate]["value"] is None
    assert rates[missing_rate]["unavailable_reason"] == "denominator is zero"
    json.dumps(report, allow_nan=False)


def test_scores_require_complete_decisions_and_both_populations() -> None:
    data = predictions(["a", "oos"], ["a", None])
    for incomplete in (True, False):
        if not incomplete:
            data = predictions(["a", "oos"], ["a", "oos"])
            data.metadata["original_status"] = "incomplete"
        result = routing_summary(data, reject_label="oos", rejection_scores={"0": 0.1, "1": 0.9})
        detection = result["detection"]
        assert isinstance(detection, dict)
        assert detection["auroc"] is None
        assert "complete" in str(detection["unavailable_reason"])
    single = routing_summary(
        predictions(["a"], ["a"]), reject_label="oos", rejection_scores={"0": 0}
    )
    assert isinstance(single["detection"], dict)
    assert single["detection"]["auroc"] is None


@pytest.mark.parametrize(
    "scores",
    [
        {},
        {"0": 1},
        {"0": 0, "1": 1, "extra": 0},
        {"0": float("nan"), "1": 1},
        {"0": float("inf"), "1": 1},
        {"0": True, "1": 1},
        {"0": 10**400, "1": 1},
    ],
)
def test_invalid_scores_fail(scores: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="rejection scores"):
        routing_summary(
            predictions(["a", "oos"], ["a", "oos"]), reject_label="oos", rejection_scores=scores
        )


def test_revalidation_and_ownership() -> None:
    data = predictions(["a", "oos"], ["a", "oos"])
    before = data.model_dump_json()
    result = routing_summary(data, reject_label="oos")
    assert data.model_dump_json() == before
    assert isinstance(result["counts"], dict)
    result["counts"]["in_scope"] = -100
    assert routing_summary(data, reject_label="oos")["counts"] != result["counts"]
    with pytest.raises(ValueError, match="reject_label"):
        routing_summary(data, reject_label="unknown")
    data.predictions.append(Prediction(id="unknown", choice="a"))
    with pytest.raises(ValueError, match="IDs"):
        routing_summary(data, reject_label="oos")


def test_missing_rows_equal_explicit_failures() -> None:
    data = predictions(["a", "oos"], [None, None])
    explicit = routing_summary(data, reject_label="oos")
    data.predictions.clear()
    assert routing_summary(data, reject_label="oos") == explicit


def test_independent_sklearn_references_and_tie_invariance() -> None:
    reference = json.loads((Path(__file__).parent / "fixtures/routing-reference.json").read_text())
    assert len(reference["cases"]) == 43
    for case in reference["cases"]:
        data = predictions(
            ["oos" if item else "a" for item in case["gold"]],
            ["oos" if item else "a" for item in case["choices"]],
        )
        scores = {str(i): value for i, value in enumerate(case["scores"])}
        report = routing_summary(data, reject_label="oos", rejection_scores=scores)
        detection = report["detection"]
        assert isinstance(detection, dict)
        for name in ("auroc", "average_precision", "fpr_at_95_recall"):
            assert detection[name] == pytest.approx(case[name], abs=1e-12)
        assert report["out_of_scope_f1"] == pytest.approx(case["out_of_scope_f1"], abs=1e-12)
        assert [row["false_positive_rate"] for row in detection["curve"]] == pytest.approx(
            case["fpr"]
        )
        assert [row["recall"] for row in detection["curve"]] == pytest.approx(case["recall"])
        data.suite.examples.reverse()
        data.predictions.reverse()
        assert (
            routing_summary(data, reject_label="oos", rejection_scores=scores)["detection"]
            == detection
        )


def test_routing_cli_and_exclusive_output(tmp_path: Path) -> None:
    data = predictions(["a", "oos"], ["a", "oos"])
    source, scores, output = (
        tmp_path / name for name in ("predictions.json", "scores.json", "report.json")
    )
    source.write_text(data.model_dump_json())
    scores.write_text('{"0":0.1,"1":0.9}')
    args = [
        "routing",
        str(source),
        "--reject-label",
        "oos",
        "--scores",
        str(scores),
        "--output",
        str(output),
    ]
    assert _cli.main(args) == 0
    assert json.loads(output.read_text())["detection"]["auroc"] == 1
    assert _cli.main(args) == 2
    output.unlink()
    for invalid in ('{"0":1,"0":0,"1":1}', '{"0":true,"1":0}', "[]", '{"0":NaN,"1":0}'):
        scores.write_text(invalid)
        assert _cli.main(args) == 2
        assert not output.exists()


def test_cli_preserves_distinct_large_integer_scores(tmp_path: Path) -> None:
    data = predictions(["a", "oos"], ["a", "oos"])
    source, score_file, output = (
        tmp_path / name for name in ("predictions.json", "scores.json", "report.json")
    )
    scores = {"0": 2**53, "1": 2**53 + 1}
    source.write_text(data.model_dump_json())
    score_file.write_text(json.dumps(scores))
    expected = routing_summary(data, reject_label="oos", rejection_scores=scores)
    assert isinstance(expected["detection"], dict)
    assert expected["detection"]["auroc"] == 1.0
    assert (
        _cli.main(
            [
                "routing",
                str(source),
                "--reject-label",
                "oos",
                "--scores",
                str(score_file),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == expected
