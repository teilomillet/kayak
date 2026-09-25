"""Challenge the support pilot's evidence, label isolation, and review fallback."""

import itertools
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from examples import evaluate_support
from kayak import Client, DecisionRequest, DecisionResult
from kayak.eval import Example, Suite, evaluate, load_report, load_suite

ROOT = Path(__file__).resolve().parents[1]


def controlled_suite(count: int) -> Suite:
    suite = load_suite(ROOT / "examples/suites/support_pilot.json")
    suite.examples = [
        Example(id=f"case-{i}", text=f"Request number {i}", label=evaluate_support.OUTCOMES[i % 4])
        for i in range(count)
    ]
    return suite


def response(request: httpx.Request) -> httpx.Response:
    # The fixture protocol encodes only a row number, never a gold label or example ID.
    content = json.loads(request.content)
    assert set(content) == {"state", "questions"}
    parsed = DecisionRequest.model_validate(content)
    index = int(parsed.state.removeprefix("Request number "))
    chosen = evaluate_support.OUTCOMES[index % 4]
    model = dict(
        id="controlled/support",
        revision="fixture-v1",
        fingerprint="no-weights",
        encoder="none",
        encoder_revision="none",
        device="none",
        dtype="none",
    )
    result = DecisionResult.model_validate(
        dict(
            model=model,
            answers={
                "intent": dict(
                    choice=chosen,
                    scores={label: float(label == chosen) for label in evaluate_support.OUTCOMES},
                    probabilities={
                        label: (2.718281828459045 if label == chosen else 1.0) / 5.718281828459045
                        for label in evaluate_support.OUTCOMES
                    },
                )
            },
            input_tokens=0,
        )
    )
    return httpx.Response(200, content=result.model_dump_json())


def test_documented_simulation_is_explicitly_not_acceptance(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.evaluate_support",
            "--simulate",
            "--output",
            str(tmp_path / "simulation"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assessment = json.loads(completed.stdout)
    assert assessment["evidence_scope"] == "integration_only"
    assert not assessment["deployment_accepted"]
    assert not assessment["provisional_gates_passed"]
    assert assessment["gates"]["accuracy"]["observed"] == 0.25
    assert assessment["gates"]["distinct_tickets"]["observed"] == 20
    assert all(row["human_confirmation_required"] for row in assessment["outcomes"])
    assert load_report(tmp_path / "simulation").status == "complete"


@pytest.mark.parametrize(
    ("mode", "seconds", "passed"),
    [("http", 2.0, True), ("http", 2.01, False), ("simulated", 0.1, False)],
)
def test_gates_require_actual_measurement_and_bound_latency(
    mode: str, seconds: float, passed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = itertools.count(step=seconds)
    monkeypatch.setattr("kayak.eval._runner.perf_counter", lambda: next(clock))
    output = tmp_path / "run"
    with Client(base_url="http://test", transport=httpx.MockTransport(response)) as client:
        evaluate(client, controlled_suite(200), output=output, config={"mode": mode})
    assessment = evaluate_support.assess(output)
    assert assessment["provisional_gates_passed"] is passed
    assert assessment["deployment_accepted"] is False


@pytest.mark.parametrize("failure", ["busy", "timeout", "authentication", "network", "malformed"])
def test_failed_review_ticket_is_retained_as_incorrect_without_retry(
    failure: str, tmp_path: Path
) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if json.loads(request.content)["state"] != "Request number 3":
            return response(request)
        if failure == "network":
            raise httpx.ConnectError("fixture", request=request)
        if failure == "malformed":
            return httpx.Response(200, json={"invalid": "response"})
        status = {"busy": 503, "timeout": 504, "authentication": 401}[failure]
        return httpx.Response(status, json={"error": {"code": failure, "message": "fixture"}})

    output = tmp_path / "run"
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        evaluate(client, controlled_suite(20), output=output, warmups=0, config={"mode": "http"})
    assert len(calls) == 20
    assessment = evaluate_support.assess(output)
    saved = load_report(output)
    assert saved.status == "failed"
    assert saved.summary["accuracy"] == 19 / 20
    outcomes = evaluate_support.review_outcomes(saved)
    assert outcomes[3] == dict(
        id="case-3",
        suggestion=None,
        review_queue="review",
        reason="inference_failed_or_missing",
        human_confirmation_required=True,
    )
    assert not assessment["provisional_gates_passed"]


def test_failed_warmup_prevents_acceptance_even_when_all_tickets_match(tmp_path: Path) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("fixture", request=request)
        return response(request)

    output = tmp_path / "run"
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        evaluate(client, controlled_suite(200), output=output, config={"mode": "http"})
    assert load_report(output).summary["accuracy"] == 1.0
    assert not evaluate_support.assess(output)["provisional_gates_passed"]


def test_repeated_text_with_unique_ids_cannot_inflate_coverage(tmp_path: Path) -> None:
    suite = controlled_suite(200)
    for index, example in enumerate(suite.examples):
        example.text = f"Request number {index % 20}"
    output = tmp_path / "duplicates"
    with Client(base_url="http://test", transport=httpx.MockTransport(response)) as client:
        evaluate(client, suite, output=output, warmups=0, config={"mode": "http"})
    assessment = evaluate_support.assess(output)
    decoded = json.loads(json.dumps(assessment))
    assert load_report(output).summary["accuracy"] == 1.0
    assert decoded["gates"]["distinct_tickets"]["observed"] == 20
    assert decoded["gates"]["minimum_tickets_per_outcome"]["observed"] == 5
    assert decoded["dataset_checks"]["status"] == "needs_review"
    assert not decoded["provisional_gates_passed"]


def test_invalid_suite_fails_before_inference_and_existing_evidence_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = controlled_suite(20)
    del suite.question.criteria["review"]
    path = tmp_path / "invalid.json"
    path.write_text(suite.model_dump_json())
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "report.json"
    marker.write_text("existing evidence")
    monkeypatch.setattr(
        sys, "argv", ["evaluate_support", "--suite", str(path), "--output", str(output)]
    )
    assert evaluate_support.main() == 2
    assert marker.read_text() == "existing evidence"
    path.write_text(controlled_suite(20).model_dump_json())
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_support", "--suite", str(path), "--simulate", "--output", str(output)],
    )
    assert evaluate_support.main() == 2
    assert marker.read_text() == "existing evidence"
