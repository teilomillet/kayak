"""External classifier arithmetic, extension boundaries, and report evidence."""

import csv
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from test_eval_metrics import classification_report
from test_eval_runner import ControlledBackend

from kayak import Choice, DecisionResult, InferenceError
from kayak.eval import (
    Metric,
    MetricInput,
    Prediction,
    PredictionSet,
    _cli,
    benchmark,
    confidence_metrics,
    default_metrics,
    evaluate,
    expected_calibration_error,
    export_predictions,
    log_loss,
    metric_input,
    reliability_bins,
    render_benchmark,
    score_metrics,
    top_k_accuracy,
    write_benchmark,
)
from kayak.eval._benchmark import accuracy_interval, mcnemar_exact


def probability_predictions() -> PredictionSet:
    report = classification_report(list("abba"), list("abab"), labels=("a", "b"))
    return PredictionSet(
        system="controlled probability fixture",
        method="fixed test predictions",
        suite=report.suite,
        predictions=[
            Prediction(id="0", choice="a", probabilities={"a": 0.8, "b": 0.2}),
            Prediction(id="1", choice="b", probabilities={"a": 0.4, "b": 0.6}),
            Prediction(id="2", choice="a", probabilities={"a": 0.7, "b": 0.3}),
            Prediction(id="3", choice="b", probabilities={"a": 0.1, "b": 0.9}),
        ],
        metadata={"evidence_kind": "mock"},
    )


def test_probability_scores_and_threshold_tradeoff_have_independent_arithmetic() -> None:
    data = metric_input(probability_predictions())
    scores = score_metrics(data, (*default_metrics(), *confidence_metrics(0.75)))
    assert scores["accuracy"]["value"] == 0.5
    assert scores["log_loss"]["value"] == pytest.approx(-math.log(0.8 * 0.6 * 0.3 * 0.1) / 4)
    assert scores["brier_score"]["value"] == pytest.approx((0.08 + 0.32 + 0.98 + 1.62) / 4)
    # Each distinct confidence occupies its own equal-width bin.
    assert scores["ece"]["value"] == pytest.approx((0.2 + 0.4 + 0.7 + 0.9) / 4)
    assert scores["selective_accuracy"]["value"] == 0.5
    assert scores["coverage"]["value"] == 0.5
    zero_accepted = score_metrics(data, confidence_metrics(1.0))
    assert zero_accepted["selective_accuracy"]["value"] is None
    assert zero_accepted["coverage"]["value"] == 0.0
    # A single bin can hide errors through cancellation; parameters matter.
    assert score_metrics(data, [expected_calibration_error(bins=1)])["ece"][
        "value"
    ] == pytest.approx(0.25)


def test_extreme_probabilities_and_bin_boundaries() -> None:
    predictions = probability_predictions()
    for row in predictions.predictions:
        row.probabilities = {"a": float(row.choice == "a"), "b": float(row.choice == "b")}
    data = metric_input(predictions)
    scores = score_metrics(data, default_metrics())
    assert scores["log_loss"]["value"] == pytest.approx(-math.log(1e-15) / 2)
    assert scores["brier_score"]["value"] == 1.0
    bins = reliability_bins(data)
    assert bins[-1]["examples"] == 4
    assert bins[-1]["mean_confidence"] == 1.0
    assert all(row["examples"] == 0 for row in bins[:-1])


def test_hard_labels_and_missing_rows_never_get_invented_probabilities() -> None:
    predictions = probability_predictions()
    for row in predictions.predictions:
        row.probabilities = None
    predictions.predictions.pop()  # Missing answer remains a selected example.
    data = metric_input(predictions)
    scores = score_metrics(data, default_metrics())
    assert scores["accuracy"]["value"] == 0.5
    for name in ("log_loss", "brier_score", "ece", "top5_accuracy"):
        assert scores[name]["value"] is None
        assert scores[name]["available_examples"] == 0
        assert scores[name]["total_examples"] == 4
    result = benchmark({"partial": predictions, "other": probability_predictions()})
    pair = result["comparisons"]
    assert isinstance(pair, list) and pair[0]["eligible"] is False
    with pytest.raises(ValueError, match="every selected"):
        reliability_bins(data)


def test_candidate_order_defines_probability_ties_and_explicit_ranking_is_preserved() -> None:
    predictions = probability_predictions()
    for row in predictions.predictions:
        row.probabilities = {"b": 0.5, "a": 0.5}  # JSON map order is not candidate order.
        row.choice = "a"
    data = metric_input(predictions)
    assert all(row.ranking == ("a", "b") for row in data.samples)
    predictions.predictions[0].choice = "b"
    predictions.predictions[0].ranking = ["b", "a"]
    assert metric_input(predictions).samples[0].ranking == ("b", "a")


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "unknown_id", "label", "sum", "keys", "nan", "negative", "ranking", "unanswered"],
)
def test_prediction_validation_rejects_ambiguous_inputs(corruption: str) -> None:
    predictions = probability_predictions()
    row = predictions.predictions[0]
    if corruption == "duplicate":
        predictions.predictions.append(row)
    elif corruption == "unknown_id":
        row.id = "unknown"
    elif corruption == "label":
        row.choice = "unknown"
    elif corruption == "sum":
        row.probabilities = {"a": 0.8, "b": 0.8}
    elif corruption == "keys":
        row.probabilities = {"a": 1.0}
    elif corruption == "nan":
        row.probabilities = {"a": float("nan"), "b": 0.2}
    elif corruption == "negative":
        row.probabilities = {"a": -0.2, "b": 1.2}
    elif corruption == "ranking":
        row.ranking = ["a", "a"]
    else:
        row.choice = None
    with pytest.raises(ValueError):
        benchmark({"bad": predictions})


def test_metric_extensions_are_immutable_versioned_and_cannot_shadow_each_other() -> None:
    def custom(data: MetricInput) -> float:
        with pytest.raises(FrozenInstanceError):
            setattr(data.samples[0], "choice", "changed")  # noqa: B010 - exercise frozen runtime guard
        return sum(row.choice != row.label for row in data.samples) / len(data.samples)

    metric = Metric(
        "error_rate", custom, "First-attempt error rate.", direction="lower", version="2"
    )
    data = metric_input(probability_predictions())
    assert score_metrics(data, [metric])["error_rate"]["value"] == 0.5
    assert score_metrics(data, [metric])["error_rate"]["version"] == "2"
    for invalid in ([metric, metric], []):
        with pytest.raises(ValueError, match="unique names"):
            score_metrics(data, invalid)
    with pytest.raises(ValueError, match="finite number"):
        score_metrics(data, [Metric("bad", lambda data: float("nan"), "invalid")])
    with pytest.raises(ValueError, match="metric names"):
        Metric("../bad", custom, "invalid")


@pytest.mark.parametrize("value", [0, -1, True, 1001])
def test_invalid_calibration_parameters(value: int) -> None:
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(bins=value)


def test_invalid_metric_parameters() -> None:
    with pytest.raises(ValueError):
        top_k_accuracy(0)
    with pytest.raises(ValueError):
        log_loss(epsilon=0.0)
    with pytest.raises(ValueError):
        confidence_metrics(float("nan"))


def test_wilson_and_exact_mcnemar_arithmetic() -> None:
    # Published Wilson formula, with z=1.959963984540054 for 95% coverage.
    interval = accuracy_interval(50, 100)
    assert interval["lower"] == pytest.approx(0.4038315303659956)
    assert interval["upper"] == pytest.approx(0.5961684696340044)
    assert accuracy_interval(0, 10)["lower"] == 0.0
    assert accuracy_interval(10, 10)["upper"] == 1.0
    assert mcnemar_exact(0, 0) == mcnemar_exact(10, 10) == 1.0
    assert mcnemar_exact(0, 5) == mcnemar_exact(5, 0) == 0.0625
    # Two-sided binomial tail: 2 * (C(10,0) + C(10,1)) / 2**10.
    assert mcnemar_exact(9, 1) == 22 / 1024
    assert 0 <= mcnemar_exact(1500, 1400) <= 1


def test_export_native_roundtrip_matches_classification_and_retains_raw_evidence(
    tmp_path: Path,
) -> None:
    fixture = classification_report(list("abba"), list("abab"), labels=("a", "b"))
    run = tmp_path / "native"
    report = evaluate(
        ControlledBackend(), fixture.suite, output=run, warmups=0, config={"evidence_kind": "mock"}
    )
    original = {name: (run / name).read_bytes() for name in ("report.json", "predictions.jsonl")}
    path = tmp_path / "external.json"
    exported = export_predictions(run, path, system="fixture", method="identity")
    assert PredictionSet.model_validate_json(path.read_bytes()) == exported
    assert exported.metadata["evidence_kind"] == "mock"
    result = benchmark({"native": run, "exported": path})
    rows = result["runs"]
    assert isinstance(rows, dict)
    assert rows["native"]["classification"]["accuracy"] == report.summary["accuracy"]
    assert rows["native"]["metrics"] == rows["exported"]["metrics"]
    assert rows["exported"]["source"]["evidence_kind"] == "mock"
    pairs = result["comparisons"]
    assert isinstance(pairs, list)
    assert pairs[0]["eligible"] is True and pairs[0]["mcnemar_exact_p"] == 1.0
    assert pairs[0]["mean_latency_speedup"] is None
    assert {name: (run / name).read_bytes() for name in original} == original
    with pytest.raises(FileExistsError):
        export_predictions(run, path, system="fixture", method="identity")
    (run / "predictions.jsonl").write_bytes(original["predictions.jsonl"] + b"\n")
    with pytest.raises(ValueError, match="prediction artifact"):
        benchmark({"corrupt": run})


@pytest.mark.parametrize("warmups,repeats,failed_call", [(1, 1, 1), (0, 2, 2)])
def test_export_preserves_failed_run_gate_with_complete_first_predictions(
    tmp_path: Path, warmups: int, repeats: int, failed_call: int
) -> None:
    class FailOnce(ControlledBackend):
        def decide(
            self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
        ) -> DecisionResult:
            result = super().decide(state=state, questions=questions)
            if len(self.calls) == failed_call:
                raise InferenceError("controlled failure outside first measured attempts")
            return result

    suite = probability_predictions().suite
    complete, failed = tmp_path / "complete", tmp_path / "failed"
    evaluate(ControlledBackend(), suite, output=complete, warmups=0)
    report = evaluate(FailOnce(), suite, output=failed, warmups=warmups, repeats=repeats)
    assert report.status == "failed"
    assert all(row.attempts[0].result is not None for row in report.observations)
    path = tmp_path / "exported.json"
    exported = export_predictions(failed, path, system="fixture", method="identity")
    assert exported.metadata["original_status"] == "failed"

    for candidate in (failed, exported, path):
        result = benchmark({"complete": complete, "candidate": candidate})
        runs, pairs = result["runs"], result["comparisons"]
        assert isinstance(runs, dict) and isinstance(pairs, list)
        assert runs["candidate"]["answered_examples"] == 4
        assert runs["candidate"]["metrics"]["accuracy"]["value"] == 0.5
        assert runs["candidate"]["accuracy_interval"] is None
        assert pairs[0]["eligible"] is False
        assert "native run is not complete" in pairs[0]["exclusions"]


def test_mismatched_protocols_are_reported_and_multiple_comparisons_are_adjusted() -> None:
    baseline = probability_predictions()
    candidate = baseline.model_copy(deep=True)
    for row, example in zip(candidate.predictions, candidate.suite.examples, strict=True):
        row.choice = example.label
    candidate.method = "different method"
    blocked = benchmark({"baseline": baseline, "candidate": candidate})
    assert isinstance(blocked["comparisons"], list)
    assert blocked["comparisons"][0]["eligible"] is False
    allowed = benchmark(
        {"baseline": baseline, "candidate": candidate, "second": candidate},
        allow_recipe_change=True,
    )
    assert isinstance(allowed["comparisons"], list)
    for pair in allowed["comparisons"]:
        assert pair["paired_outcomes"] == {
            "both_correct": 2,
            "fixed": 2,
            "regressed": 0,
            "both_wrong": 0,
        }
        assert pair["mcnemar_exact_p"] == 0.5
        assert pair["mcnemar_bonferroni_p"] == 1.0
        assert pair["comparison_family_size"] == 2
    candidate.suite.examples[0].label = "b"
    mismatch = benchmark({"baseline": baseline, "candidate": candidate}, allow_recipe_change=True)
    assert isinstance(mismatch["comparisons"], list)
    assert mismatch["comparisons"][0]["exclusions"] == ["suite examples differs"]


def test_report_bundle_and_cli_do_not_need_inference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(probability_predictions().model_dump_json())
    output = tmp_path / "report"
    assert (
        _cli.main(
            ["benchmark", str(path), "--output", str(output), "--confidence-threshold", "0.75"]
        )
        == 0
    )
    assert "benchmark.md" in capsys.readouterr().out
    result = json.loads((output / "benchmark.json").read_bytes())
    assert result["runs"]["1"]["metrics"]["coverage"]["value"] == 0.5
    markdown = (output / "benchmark.md").read_text()
    assert "execution not verified" in markdown and "Wilson 95%" in markdown
    assert "Lower-is-better" in markdown
    with (output / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(default_metrics()) + 2
    assert _cli.main(["benchmark", str(path), "--output", str(output)]) == 2
    assert json.loads((output / "benchmark.json").read_bytes()) == result
    with pytest.raises(FileExistsError):
        write_benchmark(result, output)


def test_custom_metric_cannot_rewrite_evidence_without_detection(tmp_path: Path) -> None:
    path = tmp_path / "external.json"
    path.write_text(probability_predictions().model_dump_json())

    def change_source(data: MetricInput) -> float:
        path.write_text("{}")
        return 1.0

    with pytest.raises(ValueError, match="changed during analysis"):
        benchmark(
            {"external": path}, metrics=[Metric("mutating", change_source, "Invalid fixture.")]
        )


def test_report_escapes_names_for_markdown_and_csv(tmp_path: Path) -> None:
    result = benchmark({"=1+1|<script>\n": probability_predictions()})
    write_benchmark(result, tmp_path / "safe")
    assert "<script>" not in render_benchmark(result)
    with (tmp_path / "safe/metrics.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["run"].startswith("'=1+1")


def test_example_builds_an_extensible_comparison_without_inference_imports(tmp_path: Path) -> None:
    output = tmp_path / "example"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from examples.benchmark_classifiers import run; "
            "run(Path(sys.argv[1])); "
            "assert not {'torch', 'transformers', 'fastapi'} & sys.modules.keys()",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    result = json.loads((output / "benchmark.json").read_bytes())
    assert result["runs"]["constant"]["metrics"]["accuracy"]["value"] == 0.5
    assert result["runs"]["word_overlap"]["metrics"]["accuracy"]["value"] == 1.0
    assert result["runs"]["word_overlap"]["metrics"]["example_error_cost"]["value"] == 0.0
    assert result["runs"]["word_overlap"]["metrics"]["log_loss"]["value"] is None
    assert result["comparisons"][0]["paired_outcomes"]["fixed"] == 2


def test_overlap_classifier_never_uses_the_gold_labels() -> None:
    from examples.benchmark_classifiers import demo_suite, overlap_predictions

    suite = demo_suite()
    before = overlap_predictions(suite)
    for example in suite.examples:
        example.label = "billing" if example.label == "technical" else "technical"
    assert overlap_predictions(suite).predictions == before.predictions


def test_incomplete_native_runs_and_mock_identity_remain_visible(tmp_path: Path) -> None:
    fixture = probability_predictions()
    output = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        evaluate(
            ControlledBackend(interrupt_after=1),
            fixture.suite,
            output=output,
            warmups=0,
            config={"evidence_kind": "mock"},
        )
    result = benchmark({"partial": output, "external": fixture}, allow_recipe_change=True)
    runs, pairs = result["runs"], result["comparisons"]
    assert isinstance(runs, dict) and isinstance(pairs, list)
    assert runs["partial"]["accuracy_interval"] is None
    assert pairs[0]["eligible"] is False
    assert "native run is not complete" in pairs[0]["exclusions"]
    rendered = render_benchmark(result)
    assert "mock; interrupted" in rendered
    assert "Comparison unavailable" in rendered


def test_export_command_is_explicit_and_does_not_overwrite(tmp_path: Path) -> None:
    run = tmp_path / "run"
    evaluate(ControlledBackend(), probability_predictions().suite, output=run, warmups=0)
    path = tmp_path / "external.json"
    command = [
        "export",
        str(run),
        "--output",
        str(path),
        "--system",
        "other",
        "--method",
        "identity",
    ]
    assert _cli.main(command) == 0
    original = path.read_bytes()
    assert _cli.main(command) == 2
    assert path.read_bytes() == original
