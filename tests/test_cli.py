"""Exercise the developer-facing CLI: discovery, pipes, validation, and failures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from test_contract import sample_result

from kayak import Client, DecisionRequest, DecisionResult
from kayak import __main__ as cli
from kayak.decisions import MAX_REQUEST_BYTES

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kayak", *arguments],
        cwd=ROOT,
        env={**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"},
        input=input_text,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("arguments", [(), ("--help",), ("serve", "--help"), ("info", "--help")])
def test_help_is_discoverable_and_successful(arguments: tuple[str, ...]) -> None:
    result = run_cli(*arguments)
    assert result.returncode == 0
    assert "kayak" in result.stdout
    assert not result.stderr
    if "serve" in arguments:
        assert "cache" in result.stdout and "precision" in result.stdout
        assert "8000" in result.stdout and "auto" in result.stdout
    elif "info" not in arguments:
        assert all(command in result.stdout for command in ("validate", "info", "decide", "serve"))


def test_validate_supports_files_stdin_and_pretty_json() -> None:
    source = (ROOT / "examples/decision.json").read_text()
    expected = DecisionRequest.model_validate_json(source)
    for arguments in (("validate", "-"), ("validate", "examples/decision.json", "--pretty")):
        result = run_cli(*arguments, input_text=source)
        assert result.returncode == 0
        assert not result.stderr
        assert DecisionRequest.model_validate_json(result.stdout) == expected


@pytest.mark.parametrize(
    ("body", "hint"),
    [
        ({"state": {"secret": "private"}, "questions": {}}, "serialize structured state"),
        ({"state": "text", "questions": {"q": {"type": "noul"}}}, "only Choice"),
        (
            {
                "state": "text",
                "questions": {
                    "q": {
                        "instructions": "Choose",
                        "criteria": {"billing": None},
                    }
                },
            },
            "explicit text description",
        ),
    ],
)
def test_migration_errors_are_actionable_without_echoing_state(
    body: dict[str, object], hint: str
) -> None:
    result = run_cli("validate", "-", input_text=json.dumps(body))
    assert result.returncode == 2
    assert not result.stdout
    assert hint in result.stderr
    assert "private" not in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("body", ["{", "[]", "", "null"])
def test_invalid_json_produces_no_partial_stdout(body: str) -> None:
    result = run_cli("validate", "-", input_text=body)
    assert result.returncode == 2
    assert not result.stdout
    assert "error:" in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("command", ["info", "decide"])
def test_non_ascii_api_key_is_a_plain_error(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    monkeypatch.setenv("KAYAK_TEST_NON_ASCII_KEY", "private-é")
    arguments = [command]
    if command == "decide":
        arguments.append("examples/decision.json")
    result = run_cli(*arguments, "--api-key-env", "KAYAK_TEST_NON_ASCII_KEY")
    assert result.returncode == 2
    assert not result.stdout
    assert "ASCII" in result.stderr
    assert "private" not in result.stderr and "Traceback" not in result.stderr


def test_missing_file_is_a_plain_error(tmp_path: Path) -> None:
    result = run_cli("decide", str(tmp_path / "missing.json"))
    assert result.returncode == 1
    assert not result.stdout
    assert "missing.json" in result.stderr and "Traceback" not in result.stderr


def test_request_bytes_are_bounded_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * (MAX_REQUEST_BYTES + 1))
    result = run_cli("validate", str(path))
    assert result.returncode == 2
    assert "exceeds 1 MiB" in result.stderr
    assert not result.stdout


def test_request_at_byte_limit_is_accepted(tmp_path: Path) -> None:
    raw = (ROOT / "examples/decision.json").read_bytes()
    path = tmp_path / "maximum.json"
    path.write_bytes(raw + b" " * (MAX_REQUEST_BYTES - len(raw)))
    result = run_cli("validate", str(path))
    assert result.returncode == 0
    assert DecisionRequest.model_validate_json(
        result.stdout
    ) == DecisionRequest.model_validate_json(raw)


def test_decide_rejects_invalid_input_before_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff")

    def client(*, base_url: str, api_key: str | None, timeout: float) -> Client:
        pytest.fail("invalid input must fail before opening a client")

    monkeypatch.setattr(cli, "Client", client)
    monkeypatch.setattr(sys, "argv", ["kayak", "decide", str(path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert not capsys.readouterr().out


@pytest.mark.parametrize(
    ("arguments", "hint"),
    [
        (["--port", "0"], "--port"),
        (["--batch-size", "0"], "--batch-size"),
        (["--timeout", "nan"], "--timeout"),
        (["--host", "0.0.0.0"], "KAYAK_API_KEY"),
    ],
)
def test_serve_checks_configuration_before_optional_imports(
    arguments: list[str],
    hint: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    monkeypatch.delenv("KAYAK_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["kayak", "serve", *arguments])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert hint in capsys.readouterr().err


@pytest.mark.parametrize("command", ["info", "decide"])
@pytest.mark.parametrize("status", [200, 401, 503, 504, 0])
def test_remote_commands_share_client_semantics_and_exit_codes(
    command: str,
    status: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "state": "charged twice",
                "questions": {
                    "route": {
                        "instructions": "Which team?",
                        "criteria": {"billing": "Charges", "support": "Bugs"},
                    }
                },
            }
        )
    )
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.url.path == ("/v1/model" if command == "info" else "/v1/decide")
        if status == 0:
            raise httpx.ConnectError("not connected", request=request)
        if status != 200:
            return httpx.Response(status, json={"error": {"code": "fixture", "message": "failure"}})
        value = sample_result().model if command == "info" else sample_result()
        return httpx.Response(200, content=value.model_dump_json())

    def client(*, base_url: str, api_key: str | None, timeout: float) -> Client:
        return Client(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            transport=httpx.MockTransport(respond),
        )

    monkeypatch.setattr(cli, "Client", client)
    monkeypatch.setenv("KAYAK_API_KEY", "test-key")
    monkeypatch.setattr(
        sys, "argv", ["kayak", command, *([str(path)] if command == "decide" else [])]
    )
    if status == 200:
        cli.main()
        output = capsys.readouterr()
        assert not output.err
        if command == "decide":
            assert DecisionResult.model_validate_json(output.out) == sample_result()
        else:
            assert json.loads(output.out)["id"] == "fixture"
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert not capsys.readouterr().out
        message = str(exc.value)
        assert "test-key" not in message
        assert "No automatic retry" in message if status == 0 else str(status) in message
    assert len(calls) == 1
