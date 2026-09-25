"""Adversarial checks for the evaluator, without requiring released model weights."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from benchmarks.evaluation import (
    Case,
    CaseRun,
    Protocol,
    Run,
    Sample,
    compare,
    distribution,
    read_cases,
    summarize,
)
from benchmarks.inference import DEFAULT_SUITE, Arguments, parser, sweep
from kayak.decisions import DecisionResult, ModelInfo, answer_from_scores


def trial(seconds: float = 1.0, batch: int = 1) -> Run:
    case = read_cases(DEFAULT_SUITE, "dev")[0][0]
    info = ModelInfo(
        id="test",
        revision="1",
        fingerprint="fixed",
        encoder="test",
        encoder_revision="1",
        device="cpu",
        dtype="float32",
    )
    result = DecisionResult(
        model=info,
        answers={"route": answer_from_scores(["billing", "technical"], [2.0, 1.0])},
        input_tokens=10,
    )
    return Run(
        run_id=str(uuid.uuid4()),
        created_at="fixture",
        status="passed",
        protocol=Protocol(warmups=1, repeats=3, split="dev"),
        suite_sha256="fixture",
        harness_sha256="fixture",
        source={"code": "fixture"},
        environment={"host": "fixture"},
        config={"batch_size": batch},
        model=info.model_dump(),
        memory_after_load={"process_peak_rss_bytes": 1000},
        cases=[
            CaseRun(
                case=case,
                warmups=[Sample(seconds=seconds, result=result)],
                samples=[Sample(seconds=seconds, result=result) for _ in range(3)],
                memory={"process_peak_rss_bytes": 1000},
            )
        ],
    )


def groups() -> tuple[list[Run], list[Run]]:
    return [trial() for _ in range(3)], [trial(0.8, 4) for _ in range(3)]


def test_faster_identical_results_clear_repeated_trial_gate() -> None:
    baseline, candidate = groups()
    report = compare(baseline, candidate)
    assert report["status"] == "passed"
    assert report["geomean_speedup"] == pytest.approx(1.25)
    assert report["speedup_95pct_interval"] == pytest.approx([1.25, 1.25])


@pytest.mark.parametrize("field", ["environment", "suite_sha256", "harness_sha256", "model"])
def test_incompatible_experiments_cannot_win(field: str) -> None:
    baseline, candidate = groups()
    value = "other" if field.endswith("sha256") else {"changed": "yes"}
    candidate[0] = candidate[0].model_copy(update={field: value})
    assert compare(baseline, candidate)["status"] == "rejected"


@pytest.mark.parametrize("mutation", ["failed", "missing", "duplicates", "few", "source"])
def test_invalid_trial_groups_cannot_win(mutation: str) -> None:
    baseline, candidate = groups()
    if mutation == "failed":
        candidate[0].status = "failed"
    elif mutation == "missing":
        candidate[0].cases[0].samples.pop()
    elif mutation == "duplicates":
        candidate[1] = candidate[0]
    elif mutation == "few":
        candidate.pop()
    else:
        candidate[0].source = {"code": "different"}
    assert compare(baseline, candidate)["status"] == "rejected"


def test_drift_in_any_warmup_or_repeat_cannot_be_hidden() -> None:
    baseline, candidate = groups()
    sample = candidate[-1].cases[0].warmups[0]
    assert sample.result is not None
    sample.result = sample.result.model_copy(
        update={"answers": {"route": answer_from_scores(["billing", "technical"], [2.01, 1.0])}}
    )
    assert compare(baseline, candidate)["status"] == "rejected"
    assert compare(baseline, candidate, score_atol=0.02)["status"] == "passed"
    sample.result = sample.result.model_copy(
        update={"answers": {"route": answer_from_scores(["billing", "technical"], [1.0, 2.0])}}
    )
    assert compare(baseline, candidate, score_atol=10)["status"] == "rejected"


def test_fixed_denominator_counts_failures_and_missing_samples() -> None:
    run = trial()
    run.cases[0].samples = [run.cases[0].samples[0], Sample(seconds=0.1, error="failure")]
    summary = summarize(run)
    assert summary["labels"] == 3
    assert summary["correct"] == 1
    assert summary["accuracy"] == 1 / 3
    assert summary["failed_or_missing"] == 2


def test_gold_labels_are_not_model_agreement() -> None:
    run = trial()
    run.cases[0].case.expected["route"] = "technical"
    assert summarize(run)["correct"] == 0
    assert run.status == "passed"  # Execution success is separate from task quality.


def test_memory_regression_rejects_otherwise_fast_candidate() -> None:
    baseline, candidate = groups()
    candidate[0].cases[0].memory["process_peak_rss_bytes"] = 1200
    assert compare(baseline, candidate)["status"] == "rejected"


def test_one_slow_case_is_not_hidden_by_aggregate_speedup() -> None:
    baseline, candidate = groups()
    for run in baseline + candidate:
        extra = run.cases[0].model_copy(deep=True)
        extra.case.id = "extra"
        run.cases.append(extra)
    for run in candidate:
        for sample in run.cases[1].samples:
            sample.seconds = 1.5
        for sample in run.cases[0].samples:
            sample.seconds = 0.1
    report = compare(baseline, candidate)
    assert report["status"] == "rejected"
    assert "latency regression" in str(report["reasons"])


def test_no_tail_estimate_from_a_handful_of_samples() -> None:
    assert distribution([1.0, 2.0])["p95"] is None
    assert distribution([float(i) for i in range(1, 21)])["p95"] == pytest.approx(19.05)


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_invalid_thresholds_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        compare(*groups(), score_atol=threshold)


def test_suite_rejects_duplicate_ids_and_missing_labels(tmp_path: Path) -> None:
    case = read_cases(DEFAULT_SUITE, "dev")[0][0]
    path = tmp_path / "bad.jsonl"
    path.write_text(case.model_dump_json() + "\n" + case.model_dump_json())
    with pytest.raises(ValueError, match="unique"):
        read_cases(path, "dev")
    value = case.model_dump()
    value["expected"] = {}
    with pytest.raises(ValueError, match="label every"):
        Case.model_validate(value)


def test_frozen_search_winner_requires_holdout_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks import inference

    calls: list[tuple[int, str]] = []

    def launch(args: Arguments, output: Path, batch: int, split: str) -> Run:
        calls.append((batch, split))
        run = trial(1.0 if batch == 1 else 0.8, batch)
        if split == "holdout" and batch == 4:
            run.status = "failed"
        return run

    monkeypatch.setattr(inference, "launch", launch)
    args = parser().parse_args(
        ["sweep", "--output", str(tmp_path / "sweep"), "--batch-sizes", "1", "4"],
        namespace=Arguments(),
    )
    report = sweep(args)
    assert report["status"] == "confirmation_failed"
    assert report["selected_batch_size"] == 4
    assert calls.count((1, "dev")) == calls.count((4, "dev")) == 3
    assert calls.count((1, "holdout")) == calls.count((4, "holdout")) == 3


@pytest.mark.inference
@pytest.mark.parametrize("transport", ["local", "http"])
def test_real_tiny_model_evaluation_saves_complete_evidence(
    tiny_bundle: Path,
    tmp_path: Path,
    transport: str,
) -> None:
    from benchmarks.measure import run_model

    suite = tmp_path / "suite.jsonl"
    case = read_cases(DEFAULT_SUITE, "dev")[0][0]
    suite.write_text(case.model_dump_json() + "\n")
    output = tmp_path / "results"
    run = run_model(
        suite=suite,
        output=output,
        protocol=Protocol.model_validate(
            {"transport": transport, "warmups": 1, "repeats": 2, "split": "dev"}
        ),
        model_id=str(tiny_bundle),
        device="cpu",
        dtype="float32",
        batch_size=2,
        cache_dir=None,
        local_files_only=True,
    )
    assert run.status == "passed", run.error
    restored = Run.model_validate_json((output / "run.json").read_text())
    assert restored == run
    assert len(restored.cases[0].samples) == 2
    assert restored.cases[0].samples[0].result is not None
    assert restored.memory_after_load["process_peak_rss_bytes"]
    assert restored.load_seconds is not None
    with pytest.raises(FileExistsError):
        output.mkdir(exist_ok=False)


def test_wrong_candidate_ids_are_rejected_not_treated_as_fast() -> None:
    baseline, candidate = groups()
    sample = candidate[0].cases[0].samples[0]
    assert sample.result is not None
    sample.result = sample.result.model_copy(
        update={"answers": {"route": answer_from_scores(["other", "technical"], [2.0, 1.0])}}
    )
    assert compare(baseline, candidate)["status"] == "rejected"


def test_crashed_worker_without_report_cannot_disappear(tmp_path: Path) -> None:
    from benchmarks.inference import read_runs

    (tmp_path / "trial-0.attempt.json").write_text('{"exit_code": -9}')
    with pytest.raises(ValueError, match="cannot be omitted"):
        read_runs(tmp_path)


def test_skewed_trials_cannot_pass_below_the_reported_minimum_speedup() -> None:
    baseline = [trial(seconds) for seconds in [1.0] * 5 + [4.0] * 4]
    candidate = [trial(0.96, 4) for _ in range(9)]
    report = compare(baseline, candidate)
    assert report["geomean_speedup"] == pytest.approx(1 / 0.96)
    assert report["status"] == "rejected"
    assert "aggregate speedup" in str(report["reasons"])
    interval = report["speedup_95pct_interval"]
    assert isinstance(interval, list)
    assert interval[0] == pytest.approx(1 / 0.96)
