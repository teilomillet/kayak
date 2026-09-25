"""Independent arithmetic examples and faults at the model/artifact boundaries."""

import json
import random
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from kayak import Choice, Client, DecisionRequest, DecisionResult, InferenceError, Model, ModelInfo
from kayak.decisions import answer_from_scores, request_from
from kayak.eval import Example, Suite, _runner, compare, evaluate, load_report
from kayak.eval._metrics import distribution

MODEL = ModelInfo(
    id="controlled",
    revision="1",
    fingerprint="test",
    encoder="none",
    encoder_revision="none",
    device="cpu",
    dtype="float32",
)


@pytest.fixture
def suite() -> Suite:
    return Suite(
        name="arithmetic",
        split="dev",
        question=Choice(instructions="Select intent", criteria={"a": "Alpha", "b": "Beta"}),
        examples=[
            Example(id="0", text="first", label="a"),
            Example(id="1", text="second", label="a"),
            Example(id="2", text="third", label="b"),
            Example(id="3", text="fourth", label="b"),
        ],
    )


class ControlledBackend:
    def __init__(self, *, fail: bool = False, interrupt_after: int | None = None) -> None:
        self.fail = fail
        self.interrupt_after = interrupt_after
        self.calls: list[str] = []

    def decide(
        self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        if self.interrupt_after == len(self.calls):
            raise KeyboardInterrupt
        self.calls.append(state)
        if self.fail and state == "fourth":
            raise InferenceError("private-backend-detail", request_id="test-failure")
        request = request_from(state, questions)
        keys = list(request.questions["intent"].criteria)
        scores = [2.0] + [1.0] * (len(keys) - 1)
        return DecisionResult(
            model=MODEL, answers={"intent": answer_from_scores(keys, scores)}, input_tokens=5
        )


def test_failures_stay_in_denominator_and_repeats_are_not_extra_examples(
    suite: Suite,
    tmp_path: Path,
) -> None:
    backend = ControlledBackend(fail=True)
    output = tmp_path / "result"
    report = evaluate(backend, suite, output=output, warmups=0, repeats=3)
    assert report.status == "failed"
    assert report.summary["examples"] == 4
    assert report.summary["accuracy"] == 0.5
    assert report.summary["top5_accuracy"] == 0.75
    # A: precision=2/3, recall=1, F1=.8; B: F1=0. Macro F1=.4.
    assert report.summary["macro_f1"] == pytest.approx(0.4)
    assert report.summary["failed_or_missing_examples"] == 1
    assert report.summary["failed_calls"] == 3
    assert len(backend.calls) == 12
    assert load_report(output) == report
    assert "private-backend-detail" not in (output / "predictions.jsonl").read_text()
    with pytest.raises(ValueError, match="complete runs"):
        compare(output, output)


def test_warmups_are_separate_and_execution_order_is_repeatable(
    suite: Suite,
    tmp_path: Path,
) -> None:
    backend = ControlledBackend()
    report = evaluate(backend, suite, output=tmp_path / "run", warmups=2, repeats=2, seed=17)
    expected_order = list(suite.examples)
    random.Random(17).shuffle(expected_order)
    assert backend.calls[:2] == [expected_order[0].text] * 2
    assert [row.id for row in report.observations] == [row.id for row in expected_order]
    assert len(report.warmups) == 2
    assert report.summary["accuracy"] == 0.5
    latency = report.summary["latency"]
    assert isinstance(latency, dict)
    assert latency["calls"] == 8 and latency["p95_seconds"] is None
    with pytest.raises(FileExistsError):
        evaluate(backend, suite, output=tmp_path / "run")
    assert len(backend.calls) == 10


def test_interrupt_preserves_partial_results_and_full_denominator(
    suite: Suite,
    tmp_path: Path,
) -> None:
    backend = ControlledBackend(interrupt_after=1)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        evaluate(backend, suite, output=output, warmups=0, repeats=2)
    report = load_report(output)
    assert report.status == "interrupted"
    assert len(report.observations) == 1
    assert report.summary["examples"] == 4
    assert report.summary["failed_or_missing_examples"] == 3


def test_model_identity_changes_become_failures(suite: Suite, tmp_path: Path) -> None:
    backend = ControlledBackend()

    def respond(request: httpx.Request) -> httpx.Response:
        incoming = DecisionRequest.model_validate_json(request.content)
        result = backend.decide(state=incoming.state, questions=incoming.questions)
        if len(backend.calls) > 1:
            result = result.model_copy(
                update={"model": MODEL.model_copy(update={"revision": "changed"})}
            )
        return httpx.Response(200, content=result.model_dump_json())

    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        report = evaluate(client, suite, output=tmp_path / "identity", warmups=0)
    assert report.status == "failed"
    assert report.model == MODEL
    assert report.summary["failed_or_missing_examples"] == 3


def test_http_body_has_no_gold_labels_and_uses_every_candidate(
    suite: Suite,
    tmp_path: Path,
) -> None:
    seen: list[DecisionRequest] = []
    backend = ControlledBackend()

    def respond(request: httpx.Request) -> httpx.Response:
        assert set(json.loads(request.content)) == {"state", "questions"}
        incoming = DecisionRequest.model_validate_json(request.content)
        seen.append(incoming)
        result = backend.decide(state=incoming.state, questions=incoming.questions)
        return httpx.Response(200, content=result.model_dump_json())

    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        report = evaluate(client, suite, output=tmp_path / "http", warmups=0)
    assert report.status == "complete"
    assert len(seen) == 4
    assert all(request.questions["intent"] == suite.question for request in seen)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "summary", "order", "warmup"])
def test_saved_run_corruption_cannot_be_compared(
    corruption: str,
    suite: Suite,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    evaluate(ControlledBackend(), suite, output=output)
    path = output / "predictions.jsonl"
    rows = path.read_text().splitlines()
    if corruption == "missing":
        path.write_text("\n".join(rows[:-1]) + "\n")
    elif corruption == "duplicate":
        path.write_text("\n".join([rows[0], *rows]) + "\n")
    elif corruption == "order":
        path.write_text("\n".join(reversed(rows)) + "\n")
    else:
        metadata = output / "report.json"
        payload = json.loads(metadata.read_text())
        if corruption == "summary":
            payload["summary"]["accuracy"] = 1.0
        else:
            payload["warmups"][0]["result"]["model"]["revision"] = "changed"
        metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        compare(output, output)


def test_comparison_rejects_different_cases_and_requires_explicit_recipe_change(
    suite: Suite,
    tmp_path: Path,
) -> None:
    first, second, third = (tmp_path / name for name in ("first", "second", "third"))
    evaluate(ControlledBackend(), suite, output=first)
    changed = suite.model_copy(deep=True)
    changed.question = Choice(instructions="Another instruction", criteria=suite.question.criteria)
    evaluate(ControlledBackend(), changed, output=second)
    with pytest.raises(ValueError, match="recipe changed"):
        compare(first, second)
    result = compare(first, second, allow_recipe_change=True)
    assert result["recipe_changed"] is True
    assert result["mean_latency_speedup"] is None
    changed.examples = changed.examples[:2]
    evaluate(ControlledBackend(), changed, output=third)
    with pytest.raises(ValueError, match="identical dataset"):
        compare(first, third)


def test_comparison_requires_explicit_request_transform_change(
    suite: Suite, tmp_path: Path
) -> None:
    baseline, candidate, repeated = (tmp_path / name for name in ("base", "candidate", "repeat"))
    evaluate(ControlledBackend(), suite, output=baseline)
    config: dict[str, object] = {"request_transform": "question-first-v1"}
    evaluate(ControlledBackend(), suite, output=candidate, config=config)
    evaluate(ControlledBackend(), suite, output=repeated, config=config)
    with pytest.raises(ValueError, match="recipe changed"):
        compare(baseline, candidate)
    result = compare(baseline, candidate, allow_recipe_change=True)
    assert result["recipe_changed"] is True
    assert result["baseline_request_transform"] == "identity"
    assert result["candidate_request_transform"] == "question-first-v1"
    assert result["mean_latency_speedup"] is None
    custom_timing = "custom backend execution and synchronization are not verified"
    assert result["latency_comparison_exclusions"] == ["input recipe changed", custom_timing]
    matched = compare(candidate, repeated)
    assert matched["recipe_changed"] is False
    assert matched["mean_latency_speedup"] is None
    assert matched["latency_comparison_exclusions"] == [custom_timing]


def test_memory_guard_stops_calls_without_dropping_cases(
    suite: Suite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_runner, "memory_snapshot", lambda _: {"process_peak_rss_bytes": 2**31})
    monkeypatch.setattr(_runner, "environment", lambda _: {})
    backend = Mock(spec=Model)
    backend.info = MODEL
    report = evaluate(backend, suite, output=tmp_path / "guard", max_memory_gib=1.0)
    assert report.status == "failed" and report.error == "MemoryBudgetExceeded"
    assert report.summary["failed_or_missing_examples"] == 4
    backend.decide.assert_not_called()


def test_invalid_suite_is_rejected_before_calling_or_writing(suite: Suite, tmp_path: Path) -> None:
    suite.examples[0].label = "missing"
    backend = ControlledBackend()
    with pytest.raises(ValueError):
        evaluate(backend, suite, output=tmp_path / "invalid")
    assert not backend.calls and not list(tmp_path.iterdir())


def test_newer_predictions_remain_inspectable_after_process_death(
    suite: Suite, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    evaluate(ControlledBackend(), suite, output=output)
    metadata = output / "report.json"
    payload = json.loads(metadata.read_text())
    payload.update(status="running", model=None, summary={}, warmups=[])
    metadata.write_text(json.dumps(payload))
    report = load_report(output)
    assert report.status == "running" and report.summary["accuracy"] == 0.5
    assert report.model == MODEL
    with pytest.raises(ValueError, match="complete runs"):
        compare(output, output)


def test_unavailable_memory_measurement_does_not_pass_a_budget(
    suite: Suite, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_runner, "memory_snapshot", lambda _: {})
    monkeypatch.setattr(_runner, "environment", lambda _: {})
    backend = Mock(spec=Model)
    backend.info = MODEL
    with pytest.raises(ValueError, match="cannot be enforced"):
        evaluate(backend, suite, output=tmp_path / "run", max_memory_gib=1.0)
    backend.decide.assert_not_called()
    assert load_report(tmp_path / "run").status == "interrupted"


def test_later_success_does_not_erase_first_attempt_failure(suite: Suite, tmp_path: Path) -> None:
    backend = ControlledBackend()
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls % 3 == 1:
            return httpx.Response(503, json={"error": {"code": "busy", "message": "busy"}})
        incoming = DecisionRequest.model_validate_json(request.content)
        result = backend.decide(state=incoming.state, questions=incoming.questions)
        if calls % 3 == 0:
            result = result.model_copy(
                update={"answers": {"intent": answer_from_scores(["a", "b"], [3.0, 1.0])}}
            )
        return httpx.Response(200, content=result.model_dump_json())

    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        report = evaluate(client, suite, output=tmp_path / "run", warmups=0, repeats=3)
    assert report.summary["accuracy"] == 0
    assert report.summary["failed_or_missing_examples"] == 4
    assert report.summary["failed_calls"] == 4
    assert report.summary["max_repeated_score_drift"] == 1.0
    assert calls == 12


def test_latency_sample_boundary_and_interpolation() -> None:
    assert distribution([])["mean_seconds"] is None
    assert distribution([float(value) for value in range(1, 20)])["p95_seconds"] is None
    result = distribution([float(value) for value in range(1, 21)])
    assert result["mean_seconds"] == result["median_seconds"] == 10.5
    assert result["p95_seconds"] == pytest.approx(19.05)
