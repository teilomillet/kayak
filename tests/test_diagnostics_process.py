"""Exercise CLI configuration, real HTTP, and signal-driven log shutdown."""

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import httpx
import pytest
from test_server import body

from kayak import Client, DecisionResult, RemoteError
from kayak import __main__ as cli

ROOT = Path(__file__).resolve().parents[1]
SERVER = """
import sys
from unittest.mock import patch
sys.path.insert(0, "tests")
from test_server import ControlledModel
from kayak.__main__ import main
sys.argv = ["kayak", "serve", "--port", sys.argv[1], *sys.argv[2:]]
with patch("kayak.runtime.load", return_value=ControlledModel()):
    main()
"""


@pytest.mark.parametrize("mode", [None, "off", "json"])
def test_cli_process_serves_and_flushes_diagnostics(mode: str | None, tmp_path: Path) -> None:
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    output_path, error_path = tmp_path / "stdout", tmp_path / "stderr"
    with output_path.open("w") as output, error_path.open("w") as errors:
        process = subprocess.Popen(
            [sys.executable, "-c", SERVER, str(port), *(["--diagnostics", mode] if mode else [])],
            cwd=ROOT,
            env={**os.environ, "KAYAK_API_KEY": "private-process-key"},
            stdout=output,
            stderr=errors,
            text=True,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base_url, timeout=2) as http:
                deadline = monotonic() + 10
                while True:
                    assert process.poll() is None, error_path.read_text()
                    try:
                        response = http.get("/health")
                        response.raise_for_status()
                        break
                    except httpx.ConnectError:
                        assert monotonic() < deadline, "CLI did not become ready"
                        sleep(0.05)
                assert response.json() == {"ready": True}
                response = http.post(
                    "/v1/decide",
                    json=body("private-process-input"),
                    headers={"authorization": "Bearer private-process-key"},
                )
                response.raise_for_status()
                assert DecisionResult.model_validate_json(response.content).model.id == "controlled"
                request_id = response.headers["x-request-id"]
            with Client(base_url=base_url) as client:
                with pytest.raises(RemoteError) as rejected:
                    client.model_info()
                assert rejected.value.status_code == 401
                assert rejected.value.request_id is not None
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                pytest.fail("CLI did not shut down")
    assert process.returncode in (0, -signal.SIGTERM), error_path.read_text()
    records: list[dict[str, object]] = [
        json.loads(line) for line in error_path.read_text().splitlines() if line.startswith("{")
    ]
    if mode == "json":
        assert not output_path.read_text()  # Uvicorn's ordinary access output is disabled.
        assert records[0]["event"] == "startup.started"
        assert records[-1]["event"] == "shutdown.completed"
        decision = [record for record in records if record.get("request_id") == request_id]
        assert [record["event"] for record in decision] == [
            "inference.started",
            "inference.completed",
            "http.completed",
        ]
        assert any(record.get("request_id") == rejected.value.request_id for record in records)
        assert "private-process-" not in json.dumps(records)
    else:
        assert records == []


@pytest.mark.parametrize("status", [200, 422, 503])
def test_cli_prints_request_id_on_response_errors(
    status: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def client(*, base_url: str, api_key: str | None, timeout: float) -> Client:
        return Client(
            base_url=base_url,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    status,
                    headers={"x-request-id": "request-123"},
                    json={"error": {"code": "invalid_request", "message": "failed"}},
                )
            ),
        )

    monkeypatch.setattr(cli, "Client", client)
    monkeypatch.setattr(sys, "argv", ["kayak", "info"])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    output = capsys.readouterr()
    assert not output.out
    assert "[request_id=request-123]" in str(failure.value) + output.err


def test_invalid_diagnostics_option_fails_before_loading() -> None:
    with pytest.raises(SystemExit) as failure:
        cli.argument_parser().parse_args(["serve", "--diagnostics", "verbose"])
    assert failure.value.code == 2
