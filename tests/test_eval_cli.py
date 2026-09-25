"""CLI validation precedes effects; real inference crosses both public interfaces."""

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from test_eval_runner import ControlledBackend

from kayak import Choice, Client, DecisionRequest, Model
from kayak.eval import Example, Suite, _cli, evaluate, load_report


def small_suite() -> Suite:
    return Suite(
        name="cli-fixture",
        split="dev",
        question=Choice(instructions="Which team?", criteria={"a": "billing", "b": "technical"}),
        examples=[
            Example(id="first", text="charged twice", label="a"),
            Example(id="second", text="service outages", label="b"),
        ],
        provenance={"coverage": "subset"},
    )


@pytest.mark.parametrize("arguments", [["kayak", "eval"], ["kayak.eval"]])
def test_both_entry_points_discover_commands(arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", *arguments, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert all(word in result.stdout for word in ("prepare", "run", "compare", "history"))
    assert not result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["--repeats", "0"],
        ["--timeout", "nan"],
        ["--min-accuracy", "nan"],
        ["--max-memory-gib", "inf"],
        ["--base-url", "http://test", "--device", "mps"],
        ["--base-url", "http://test", "--max-memory-gib", "1"],
    ],
)
def test_bad_configuration_fails_before_data_or_model_access(
    arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(**kwargs: object) -> Suite:
        pytest.fail("invalid configuration reached data loading")

    monkeypatch.setattr(_cli, "banking77", unexpected)
    assert _cli.main(["run", "banking77", "--output", str(tmp_path / "run"), *arguments]) == 2
    assert not list(tmp_path.iterdir())


def test_http_cli_accuracy_target_and_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = ControlledBackend()

    def respond(request: httpx.Request) -> httpx.Response:
        incoming = DecisionRequest.model_validate_json(request.content)
        result = backend.decide(state=incoming.state, questions=incoming.questions)
        return httpx.Response(200, content=result.model_dump_json())

    def client(**kwargs: object) -> Client:
        return Client(base_url="http://test", transport=httpx.MockTransport(respond))

    monkeypatch.setattr(_cli, "Client", client)
    monkeypatch.setattr(_cli, "banking77", lambda **kwargs: small_suite())
    output = tmp_path / "run"
    code = _cli.main(
        [
            "run",
            "banking77",
            "--base-url",
            "http://test",
            "--output",
            str(output),
            "--min-accuracy",
            "0.75",
            "--warmups",
            "0",
        ]
    )
    assert code == 1  # Execution completed, but the requested quality target was missed.
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete" and summary["accuracy"] == 0.5
    assert summary["weighted_f1"] == pytest.approx(1 / 3)
    assert summary["matthews_correlation"] == 0.0
    assert load_report(output).transport == "http"
    assert _cli.main(["history", str(tmp_path)]) == 0
    history = json.loads(capsys.readouterr().out)
    assert history["accuracy"] == history["balanced_accuracy"] == 0.5
    assert history["weighted_f1"] == pytest.approx(1 / 3)
    assert history["zero_recall_labels"] == ["b"]
    assert _cli.main(["compare", str(output), str(output)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["quality_delta"]["accuracy"] == 0
    assert comparison["quality_delta"]["matthews_correlation"] == 0
    assert comparison["paired_outcomes"] == {
        "both_correct": 1,
        "fixed": 0,
        "regressed": 0,
        "both_wrong": 1,
    }
    assert comparison["mean_latency_speedup"] is None


@pytest.mark.inference
def test_actual_tiny_model_local_and_live_http(tiny_model: Model, tmp_path: Path) -> None:
    import uvicorn

    from kayak.server import create_app

    suite = small_suite()
    local = evaluate(tiny_model, suite, output=tmp_path / "local", warmups=1, repeats=2)
    assert local.status == "complete" and local.transport == "local"
    assert load_report(tmp_path / "local") == local
    server = uvicorn.Server(uvicorn.Config(create_app(lambda: tiny_model), log_level="error"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            with Client(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}") as client:
                remote = evaluate(client, suite, output=tmp_path / "http", warmups=1, repeats=2)
            assert remote.status == "complete" and remote.transport == "http"
            assert load_report(tmp_path / "http") == remote
            for direct, http in zip(local.observations, remote.observations, strict=True):
                assert direct.id == http.id
                assert [attempt.result for attempt in direct.attempts] == [
                    attempt.result for attempt in http.attempts
                ]
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()
    assert tiny_model._encoder is None
