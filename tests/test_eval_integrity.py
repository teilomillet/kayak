"""Attack saved evidence and timing boundaries with controlled backends only."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_eval_runner import MODEL, ControlledBackend

from kayak import Choice, DecisionRequest, DecisionResult, InferenceError, Model
from kayak.decisions import answer_from_scores
from kayak.eval import Example, Suite, _runner, evaluate, load_report, summarize


@pytest.fixture
def integrity_suite() -> Suite:
    return Suite(
        name="integrity-fixture",
        split="dev",
        question=Choice(instructions="Select intent", criteria={"a": "Alpha", "b": "Beta"}),
        examples=[
            Example(id="0", text="café", label="a"),
            Example(id="1", text="second", label="a"),
            Example(id="2", text="third", label="b"),
            Example(id="3", text="fourth", label="b"),
        ],
    )


@pytest.fixture
def complete_run(integrity_suite: Suite, tmp_path: Path) -> Path:
    output = tmp_path / "run"
    evaluate(ControlledBackend(), integrity_suite, output=output)
    return output


def test_new_runs_bind_exact_prediction_bytes(complete_run: Path) -> None:
    raw = (complete_run / "predictions.jsonl").read_bytes()
    report = load_report(complete_run)
    assert report.schema_version == 3
    assert report.prediction_artifact is not None
    assert report.prediction_artifact.bytes == len(raw)
    assert report.prediction_artifact.sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("corruption", ["tokens", "scores", "append", "truncate"])
def test_changed_raw_evidence_is_rejected_even_when_quality_is_unchanged(
    complete_run: Path, corruption: str
) -> None:
    path = complete_run / "predictions.jsonl"
    raw = path.read_bytes()
    if corruption == "tokens":
        changed = raw.replace(b'"input_tokens":5', b'"input_tokens":9', 1)
        assert len(changed) == len(raw)
    elif corruption == "scores":
        rows = raw.splitlines(keepends=True)
        row = json.loads(rows[0])
        row["attempts"][0]["result"]["answers"]["intent"] = answer_from_scores(
            ["a", "b"], [200.0, 100.0]
        ).model_dump()
        rows[0] = (json.dumps(row) + "\n").encode()
        changed = b"".join(rows)
    elif corruption == "append":
        changed = raw + raw.splitlines(keepends=True)[-1]
    else:
        changed = b"".join(raw.splitlines(keepends=True)[:-1])
    assert changed != raw
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="prediction artifact"):
        load_report(complete_run)


def test_running_checkpoint_verifies_its_prefix_and_retains_later_rows(
    integrity_suite: Suite, tmp_path: Path
) -> None:
    integrity_suite.examples = [
        Example(id=str(index), text=f"ticket {index}", label="a") for index in range(26)
    ]
    output = tmp_path / "run"
    checkpoint: bytes | None = None

    def capture_checkpoint(done: int, total: int) -> None:
        nonlocal checkpoint
        if done == total:
            checkpoint = (output / "report.json").read_bytes()

    evaluate(ControlledBackend(), integrity_suite, output=output, progress=capture_checkpoint)
    assert checkpoint is not None
    (output / "report.json").write_bytes(checkpoint)
    report = load_report(output)
    raw = (output / "predictions.jsonl").read_bytes()
    assert report.status == "running" and len(report.observations) == 26
    assert report.prediction_artifact is not None
    assert 0 < report.prediction_artifact.bytes < len(raw)

    # Newer rows are recoverable, but never promote the checkpoint to complete.
    committed = raw[: report.prediction_artifact.bytes]
    tail = raw[report.prediction_artifact.bytes :]
    changed_tail = tail.replace(b'"input_tokens":5', b'"input_tokens":9', 1)
    assert changed_tail != tail
    (output / "predictions.jsonl").write_bytes(committed + changed_tail)
    assert load_report(output).status == "running"

    changed_prefix = committed.replace(b'"input_tokens":5', b'"input_tokens":9', 1)
    (output / "predictions.jsonl").write_bytes(changed_prefix + tail)
    with pytest.raises(ValueError, match="artifact hash"):
        load_report(output)


def test_initial_checkpoint_recovers_rows_without_synthesizing_integrity(
    integrity_suite: Suite, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    checkpoint: bytes | None = None

    def capture_checkpoint(done: int, total: int) -> None:
        nonlocal checkpoint
        if done == 1:
            checkpoint = (output / "report.json").read_bytes()

    evaluate(ControlledBackend(), integrity_suite, output=output, progress=capture_checkpoint)
    assert checkpoint is not None
    (output / "report.json").write_bytes(checkpoint)
    report = load_report(output)
    assert report.status == "running" and report.model == MODEL
    assert report.prediction_artifact is not None and report.prediction_artifact.bytes == 0
    assert len(report.observations) == 4 and not report.warmups


@pytest.mark.parametrize("corruption", ["partial_tail", "split_checkpoint"])
def test_running_artifacts_require_complete_rows(complete_run: Path, corruption: str) -> None:
    metadata = complete_run / "report.json"
    payload = json.loads(metadata.read_bytes())
    raw = (complete_run / "predictions.jsonl").read_bytes()
    payload["status"] = "running"
    size = 0 if corruption == "partial_tail" else 1
    payload["prediction_artifact"] = {
        "sha256": hashlib.sha256(raw[:size]).hexdigest(),
        "bytes": size,
    }
    metadata.write_text(json.dumps(payload))
    if corruption == "partial_tail":
        (complete_run / "predictions.jsonl").write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="row"):
        load_report(complete_run)


@pytest.mark.parametrize("version", [1, 2])
def test_older_artifacts_keep_their_original_summary_contract(
    complete_run: Path, version: int
) -> None:
    metadata = complete_run / "report.json"
    payload = json.loads(metadata.read_bytes())
    payload["summary"] = summarize(load_report(complete_run), legacy=True)
    payload["schema_version"] = version
    if version == 1:
        del payload["prediction_artifact"]
    metadata.write_text(json.dumps(payload))
    original = metadata.read_bytes()
    report = load_report(complete_run)
    assert report.schema_version == version
    assert (report.prediction_artifact is None) == (version == 1)
    assert report.status == "complete" and report.summary["accuracy"] == 0.5
    assert "weighted_f1" not in report.summary
    assert summarize(report)["weighted_f1"] == pytest.approx(1 / 3)
    assert metadata.read_bytes() == original
    payload["summary"]["accuracy"] = 1.0
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary does not match"):
        load_report(complete_run)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_integrity_declaration_must_match_schema_version(complete_run: Path, version: int) -> None:
    metadata = complete_run / "report.json"
    payload = json.loads(metadata.read_bytes())
    payload["schema_version"] = version
    if version >= 2:
        del payload["prediction_artifact"]
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prediction artifact"):
        load_report(complete_run)


@pytest.mark.parametrize("corruption", ["missing_metric", "wrong_metric", "extra_metric"])
def test_expanded_summary_is_verified_exactly(complete_run: Path, corruption: str) -> None:
    metadata = complete_run / "report.json"
    payload = json.loads(metadata.read_bytes())
    if corruption == "missing_metric":
        del payload["summary"]["weighted_f1"]
    elif corruption == "wrong_metric":
        payload["summary"]["matthews_correlation"] = 1.0
    else:
        payload["summary"]["unexpected_metric"] = 0.0
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary does not match"):
        load_report(complete_run)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
def test_invalid_configuration_fails_before_calls_or_output(
    integrity_suite: Suite, tmp_path: Path, value: object
) -> None:
    backend = ControlledBackend()
    with pytest.raises(ValueError, match="finite JSON"):
        evaluate(backend, integrity_suite, output=tmp_path / "run", config={"nested": [value]})
    assert not backend.calls and not list(tmp_path.iterdir())


def test_nested_configuration_is_snapshotted_before_backend_calls(
    integrity_suite: Suite, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings: dict[str, object] = {"batch_size": 1}
    config: dict[str, object] = {"settings": settings}
    backend = ControlledBackend()
    original_decide = backend.decide

    def mutate(
        *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        settings["batch_size"] = 999
        return original_decide(state=state, questions=questions)

    monkeypatch.setattr(backend, "decide", mutate)
    report = evaluate(backend, integrity_suite, output=tmp_path / "run", config=config)
    assert settings["batch_size"] == 999
    assert report.config["settings"] == {"batch_size": 1}
    assert load_report(tmp_path / "run").config == report.config


@pytest.mark.parametrize("field", ["environment", "config", "summary", "memory"])
def test_invalid_saved_measurement_metadata_is_rejected(complete_run: Path, field: str) -> None:
    metadata = complete_run / "report.json"
    payload = json.loads(metadata.read_bytes())
    payload[field]["invalid"] = -1 if field == "memory" else float("nan")
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_report(complete_run)


@pytest.mark.parametrize(
    "corruption",
    ["false_failure", "false_interruption", "excess_warmups", "missing_warmups", "partial_rows"],
)
def test_impossible_terminal_histories_are_rejected(complete_run: Path, corruption: str) -> None:
    metadata = complete_run / "report.json"
    payload = json.loads(metadata.read_bytes())
    if corruption == "false_failure":
        payload["status"] = "failed"
        message = "no recorded failure"
    elif corruption == "false_interruption":
        payload["status"] = "interrupted"
        message = "no recorded interruption"
    elif corruption == "excess_warmups":
        payload["status"] = "failed"
        payload["protocol"]["warmups"] = 0
        message = "too many warmup"
    elif corruption == "missing_warmups":
        payload["status"] = "interrupted"
        payload["error"] = "KeyboardInterrupt"
        payload["protocol"]["warmups"] = 2
        message = "before the declared warmups"
    else:
        payload["status"] = "interrupted"
        payload["error"] = "KeyboardInterrupt"
        payload["protocol"]["repeats"] = 2
        message = "only the final prediction row"
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_report(complete_run)


@pytest.mark.parametrize("run_error", [None, ""])
def test_call_failure_does_not_explain_a_missing_execution_tail(
    integrity_suite: Suite, tmp_path: Path, run_error: str | None
) -> None:
    output = tmp_path / "run"
    evaluate(ControlledBackend(fail=True), integrity_suite, output=output)
    metadata = output / "report.json"
    payload = json.loads(metadata.read_bytes())
    path = output / "predictions.jsonl"
    raw = b"".join(path.read_bytes().splitlines(keepends=True)[:-1])
    path.write_bytes(raw)
    payload["prediction_artifact"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    payload["error"] = run_error
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unfinished terminal run"):
        load_report(output)


def test_custom_memory_budget_is_rejected_before_effects(
    integrity_suite: Suite, tmp_path: Path
) -> None:
    backend = ControlledBackend()
    with pytest.raises(ValueError, match="local Model"):
        evaluate(backend, integrity_suite, output=tmp_path / "run", max_memory_gib=1.0)
    assert not backend.calls and not list(tmp_path.iterdir())


@pytest.mark.parametrize(("warmups", "repeats"), [(0, 1), (2, 3)])
@pytest.mark.parametrize("interrupt_after", [0, 1, 2, 3, 4, 5, 13, 14])
def test_interruptions_at_warmup_and_repeat_boundaries_remain_loadable(
    integrity_suite: Suite,
    tmp_path: Path,
    warmups: int,
    repeats: int,
    interrupt_after: int,
) -> None:
    backend = ControlledBackend(interrupt_after=interrupt_after)
    total_calls = warmups + len(integrity_suite.examples) * repeats
    if interrupt_after < total_calls:
        with pytest.raises(KeyboardInterrupt):
            evaluate(
                backend, integrity_suite, output=tmp_path / "run", warmups=warmups, repeats=repeats
            )
        expected_status = "interrupted"
    else:
        evaluate(
            backend, integrity_suite, output=tmp_path / "run", warmups=warmups, repeats=repeats
        )
        expected_status = "complete"
    report = load_report(tmp_path / "run")
    assert report.status == expected_status
    assert report.summary["examples"] == len(integrity_suite.examples)


@pytest.mark.parametrize("stop_after", [0, 1, 2, 3, 5, 6])
def test_memory_stops_at_warmup_and_repeat_boundaries_remain_loadable(
    integrity_suite: Suite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_after: int,
) -> None:
    controlled = ControlledBackend()
    backend = Mock(spec=Model)
    backend.info = MODEL
    backend.decide.side_effect = controlled.decide

    def memory_snapshot(device: str | None) -> dict[str, int]:
        return {"process_peak_rss_bytes": 2**31 if len(controlled.calls) >= stop_after else 0}

    monkeypatch.setattr(_runner, "environment", lambda _: {})
    monkeypatch.setattr(_runner, "memory_snapshot", memory_snapshot)
    report = evaluate(
        backend,
        integrity_suite,
        output=tmp_path / "run",
        warmups=2,
        repeats=3,
        max_memory_gib=1.0,
    )
    assert len(controlled.calls) == stop_after
    assert report.status == "failed" and report.error == "MemoryBudgetExceeded"
    assert load_report(tmp_path / "run") == report


@pytest.mark.parametrize("call_fails", [False, True])
@pytest.mark.parametrize("completion_fails", [False, True])
def test_timing_excludes_copy_and_waits_for_completion_after_failures(
    integrity_suite: Suite,
    monkeypatch: pytest.MonkeyPatch,
    call_fails: bool,
    completion_fails: bool,
) -> None:
    clock = 0.0
    synchronizations = 0
    original_copy = DecisionRequest.model_copy

    def copy_request(request: DecisionRequest, *, deep: bool = False) -> DecisionRequest:
        nonlocal clock
        clock += 100.0
        return original_copy(request, deep=deep)

    def sync() -> None:
        nonlocal clock, synchronizations
        clock += 2.0
        synchronizations += 1
        if synchronizations == 2 and completion_fails:
            raise RuntimeError("completion failed")

    backend = ControlledBackend()
    original_decide = backend.decide

    def decide(
        *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        nonlocal clock
        clock += 3.0
        if call_fails:
            raise InferenceError("call failed", request_id="original-call")
        return original_decide(state=state, questions=questions)

    monkeypatch.setattr(DecisionRequest, "model_copy", copy_request)
    monkeypatch.setattr(backend, "decide", decide)
    monkeypatch.setattr(_runner, "perf_counter", lambda: clock)
    attempt = _runner._attempt(
        backend, integrity_suite.question, integrity_suite.examples[0], MODEL, sync
    )
    assert synchronizations == 2 and attempt.seconds == 5.0
    if call_fails:
        assert attempt.error is not None
        assert (
            attempt.error.type == "InferenceError" and attempt.error.request_id == "original-call"
        )
    elif completion_fails:
        assert attempt.error is not None and attempt.error.type == "RuntimeError"
    else:
        assert attempt.result is not None


def test_interrupt_still_waits_for_completion_and_propagates(
    integrity_suite: Suite, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    backend = ControlledBackend(interrupt_after=0)

    def clock() -> float:
        events.append("clock")
        return 0.0

    monkeypatch.setattr(_runner, "perf_counter", clock)
    with pytest.raises(KeyboardInterrupt):
        _runner._attempt(
            backend,
            integrity_suite.question,
            integrity_suite.examples[0],
            MODEL,
            lambda: events.append("sync"),
        )
    assert events == ["sync", "clock", "sync", "clock"]


def test_custom_synchronization_is_recorded(integrity_suite: Suite, tmp_path: Path) -> None:
    report = evaluate(
        ControlledBackend(), integrity_suite, output=tmp_path / "run", sync=lambda: None
    )
    assert report.environment["synchronization"] == "custom"
