"""Call a real loopback service from separate CLI processes without a model download."""

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from test_cli import run_cli
from test_server import ControlledModel

from kayak import DecisionResult, ModelInfo, RankingResult
from kayak.server import create_app


def test_cli_inspection_and_decision_over_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ControlledModel()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(lambda: model, api_key="fixture-secret"),
            log_level="error",
        )
    )
    monkeypatch.setenv("KAYAK_TEST_SERVICE_KEY", "fixture-secret")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{sock.getsockname()[1]}"
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.01)
            connection = ("--base-url", address, "--api-key-env", "KAYAK_TEST_SERVICE_KEY")
            info = run_cli("info", *connection)
            assert info.returncode == 0, info.stderr
            assert ModelInfo.model_validate_json(info.stdout) == model.info
            source = Path(__file__).resolve().parents[1] / "examples/decision.json"
            result = run_cli("decide", "-", *connection, input_text=source.read_text())
            assert result.returncode == 0, result.stderr
            decision = DecisionResult.model_validate_json(result.stdout)
            assert decision.model == model.info
            assert decision.answers["department"].choice == "billing"
            assert list(decision.answers["department"].scores) == list(
                json.loads(source.read_text())["questions"]["department"]["criteria"]
            )
            result = run_cli("rank", "examples/ranking.json", *connection)
            assert result.returncode == 0, result.stderr
            ranked = RankingResult.model_validate_json(result.stdout)
            assert ranked.model == model.info
            assert [item.id for item in ranked.ranked] == ["refunds", "passwords", "exports"]
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), "service did not drain and close"
    assert model.closed
