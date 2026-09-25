"""Run the documented programs through their public entry points."""

import asyncio
import io
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

import kayak
from examples import (
    async_decisions,
    http_client,
    local_decisions,
    mock_integration,
    process_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.inference
def test_async_example_uses_the_actual_tiny_model(
    tiny_model: kayak.Model,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from kayak.server import create_app

    app = create_app(lambda: tiny_model)

    def client(*, base_url: str, api_key: str | None) -> kayak.AsyncClient:
        return kayak.AsyncClient(
            base_url=base_url, api_key=api_key, transport=httpx.ASGITransport(app)
        )

    monkeypatch.setattr(async_decisions, "AsyncClient", client)
    monkeypatch.delenv("KAYAK_API_KEY", raising=False)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            assert await async_decisions.main() == 0

    asyncio.run(scenario())
    results = [
        kayak.RankingResult.model_validate_json(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert len(results) == 2
    for result in results:
        assert result.model.id == "tests/tiny-clm"
        assert {item.id for item in result.ranked} == async_decisions.ACTIONS.keys()
        assert result.input_tokens > 0


def test_simulated_example_needs_no_optional_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "runpy.run_module('examples.mock_integration', run_name='__main__'); "
            "assert not {'torch', 'transformers', 'fastapi'} & sys.modules.keys()",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Simulated integration passed" in completed.stdout


@pytest.mark.parametrize("status", [200, 503, 504, 401, 0])
def test_http_example_handles_success_and_failure_without_retry(
    status: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if status == 0:
            raise httpx.ConnectError("unavailable", request=request)
        if status == 200:
            return mock_integration.simulated_response(request)
        return httpx.Response(status, json={"error": {"code": "fixture", "message": "failure"}})

    original = kayak.Client

    def client(*, base_url: str, api_key: str | None, timeout: float) -> kayak.Client:
        return original(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            transport=httpx.MockTransport(respond),
        )

    monkeypatch.setattr(kayak, "Client", client)
    monkeypatch.setenv("KAYAK_BASE_URL", "http://example.test")
    monkeypatch.setenv("KAYAK_API_KEY", "example-key")
    assert http_client.main() == (0 if status == 200 else 1)
    output = capsys.readouterr()
    assert len(calls) == 1
    assert calls[0].headers["authorization"] == "Bearer example-key"
    if status == 200:
        assert "Selected department: billing" in output.out
    elif status == 504:
        assert "may still be running" in output.err
    else:
        assert output.err


@pytest.mark.inference
def test_local_example_reuses_actual_tiny_model(
    tiny_bundle: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KAYAK_MODEL", str(tiny_bundle))
    monkeypatch.setenv("KAYAK_DEVICE", "cpu")
    local_decisions.main()
    output = capsys.readouterr().out
    assert "Device: cpu" in output
    assert output.count("department:") == output.count("urgency:") == 2


@pytest.mark.inference
def test_jsonl_example_preserves_records(
    tiny_bundle: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KAYAK_MODEL", str(tiny_bundle))
    monkeypatch.setenv("KAYAK_DEVICE", "cpu")
    monkeypatch.setattr(sys, "argv", ["process_jsonl", str(ROOT / "examples/tickets.jsonl")])
    process_jsonl.main()
    records = [
        process_jsonl.ClassifiedTicket.model_validate_json(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert [record.id for record in records] == ["ticket-001", "ticket-002"]
    for record in records:
        assert record.decision.model.id == "tests/tiny-clm"
        assert list(record.decision.answers) == ["department"]
        assert list(record.decision.answers["department"].scores) == ["billing", "technical"]
        assert record.decision.input_tokens > 0


def test_jsonl_reports_bad_line_after_preserving_valid_record(tmp_path: Path) -> None:
    source = tmp_path / "tickets.jsonl"
    source.write_text('{"id":"first","text":"valid"}\n{"id":"missing-text"}\n')
    with source.open() as stream:
        tickets = process_jsonl.read_tickets(stream)
        assert next(tickets).id == "first"
        with pytest.raises(ValueError, match=r"tickets.jsonl:2: invalid ticket"):
            next(tickets)


@given(identifier=st.text(min_size=1), text=st.text(min_size=1, max_size=1000))
def test_jsonl_preserves_unicode_and_embedded_newlines(identifier: str, text: str) -> None:
    ticket = process_jsonl.Ticket(id=identifier, text=text)
    stream = io.StringIO(ticket.model_dump_json() + "\n")
    assert list(process_jsonl.read_tickets(stream)) == [ticket]
