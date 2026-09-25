"""Run the application-owned Ollama recipe without a live service or model."""

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from examples import rag_quickstart


@pytest.fixture(autouse=True)
def block_live_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise AssertionError("These tests must use MockTransport, never a live service")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject)


def mock_ollama(
    monkeypatch: pytest.MonkeyPatch, respond: Callable[[httpx.Request], httpx.Response]
) -> tuple[list[httpx.Client], list[httpx.Request]]:
    real_client = httpx.Client
    clients: list[httpx.Client] = []
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    def client(*, base_url: str, timeout: float) -> httpx.Client:
        result = real_client(
            base_url=base_url, timeout=timeout, transport=httpx.MockTransport(record)
        )
        clients.append(result)
        return result

    monkeypatch.setattr(httpx, "Client", client)
    return clients, requests


@pytest.mark.parametrize(
    ("answer", "matched"), [("30 days", True), ("90 days", False), (" 30 days\n", True)]
)
def test_quickstart_sends_retrieved_context_and_checks_the_actual_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], answer: str, matched: bool
) -> None:
    clients, requests = mock_ollama(
        monkeypatch,
        lambda _: httpx.Response(
            200, json={"response": answer, "done": True, "model": "gemma3:1b"}
        ),
    )
    rag_quickstart.main()
    assert capsys.readouterr().out == (
        f"Answer: {answer}\nExpected source in context: True\nAnswer matches reference: {matched}\n"
    )
    assert len(clients) == len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://localhost:11434/api/generate"
    assert json.loads(requests[0].content) == {
        "model": "gemma3:1b",
        "prompt": (
            "Context:\n[retention]\nAudit logs are kept for 30 days.\n"
            "Question: How long are audit logs kept?\nAnswer only as '<number> days'."
        ),
        "stream": False,
    }
    assert clients[0].timeout.read == 120 and clients[0].is_closed


def test_changed_retrieval_reaches_the_prompt_and_source_assessment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "How long are audit logs kept? 90 days.\nRévisé — 日本語.\n"
    monkeypatch.setattr(
        rag_quickstart, "DOCUMENTS", {"retention": "Audit logs.", "replacement": text}
    )
    clients, requests = mock_ollama(
        monkeypatch, lambda _: httpx.Response(200, json={"response": "90 days", "done": True})
    )
    rag_quickstart.main()
    assert capsys.readouterr().out.splitlines() == [
        "Answer: 90 days",
        "Expected source in context: False",
        "Answer matches reference: False",
    ]
    assert len(requests) == 1
    prompt = json.loads(requests[0].content)["prompt"]
    assert f"[replacement]\n{text}" in prompt
    assert "[retention]" not in prompt
    assert clients[0].is_closed


@pytest.mark.parametrize("documents", [{}, {"unrelated": "Reset password from Settings."}])
def test_empty_or_zero_overlap_retrieval_stops_before_client_construction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    documents: dict[str, str],
) -> None:
    constructions: list[str] = []

    def unexpected_client(*, base_url: str, timeout: float) -> httpx.Client:
        constructions.append(base_url)
        raise AssertionError("No evidence must stop before client construction")

    monkeypatch.setattr(rag_quickstart, "DOCUMENTS", documents)
    monkeypatch.setattr(httpx, "Client", unexpected_client)
    with pytest.raises(ValueError, match="No evidence retrieved"):
        rag_quickstart.main()
    assert constructions == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("problem", ["unavailable", "status", "redirect"])
def test_provider_failures_propagate_without_retry_and_close_the_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], problem: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if problem == "unavailable":
            raise httpx.ConnectError("Service unavailable", request=request)
        if problem == "redirect":
            return httpx.Response(307, headers={"location": "http://elsewhere/api/generate"})
        return httpx.Response(503, text="Service unavailable")

    clients, requests = mock_ollama(monkeypatch, respond)
    expected_error = httpx.ConnectError if problem == "unavailable" else httpx.HTTPStatusError
    with pytest.raises(expected_error):
        rag_quickstart.main()
    assert capsys.readouterr().out == ""
    assert len(clients) == len(requests) == 1 and clients[0].is_closed


@pytest.mark.parametrize(
    "content",
    [
        b"not JSON",
        b'{"response":"30 days"}',
        b'{"done":true}',
        b'{"response":123,"done":true}',
        b'{"response":"30 days","done":"true"}',
        b'{"response":"30 days","done":1}',
    ],
)
def test_invalid_generation_records_propagate_and_close_the_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], content: bytes
) -> None:
    clients, requests = mock_ollama(monkeypatch, lambda _: httpx.Response(200, content=content))
    with pytest.raises(ValidationError):
        rag_quickstart.main()
    assert capsys.readouterr().out == ""
    assert len(clients) == len(requests) == 1 and clients[0].is_closed


@pytest.mark.parametrize(("answer", "done"), [("30 days", False), ("", True), (" \n", True)])
def test_incomplete_or_blank_answers_are_rejected_and_close_the_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
    done: bool,
) -> None:
    clients, requests = mock_ollama(
        monkeypatch, lambda _: httpx.Response(200, json={"response": answer, "done": done})
    )
    with pytest.raises(ValueError, match="incomplete or blank"):
        rag_quickstart.main()
    assert capsys.readouterr().out == ""
    assert len(clients) == len(requests) == 1 and clients[0].is_closed
