"""Retrieved context reaches native judgments through the public HTTP client."""

import json
import math
from pathlib import Path

import httpx
import pytest
from test_contract import sample_result

from examples import rag_decisions
from kayak import (
    ChoiceAnswer,
    Client,
    DecisionRequest,
    DecisionResult,
    JudgmentResult,
    NoulAnswer,
    RemoteError,
    ScoreAnswer,
)
from kayak.decisions import answer_from_scores


@pytest.mark.parametrize("selected", ["30_days", "90_days", "unknown"])
def test_retrieved_context_and_native_answers_survive_the_http_boundary(
    selected: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = {
        "rétention": "  Audit logs: 30 days.\n日本語の原文を保持。\n",
        "exceptions": "No exception stated.\nKeep this second source intact.",
    }
    queries: list[str] = []
    calls: list[httpx.Request] = []
    wire_result = DecisionResult(
        model=sample_result().model,
        input_tokens=31,
        answers={
            "retention": answer_from_scores(
                ["30_days", "90_days", "unknown"],
                [5.0 if key == selected else 0.0 for key in ("30_days", "90_days", "unknown")],
            ),
            "stated": answer_from_scores(["false", "true"], [0.0, 2.0]),
            "completeness": answer_from_scores(["0", "1", "2"], [0.0, 1.0, 3.0]),
        },
    )

    def retrieve(query: str) -> dict[str, str]:
        queries.append(query)
        return documents

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=wire_result.model_dump_json())

    def client(*, base_url: str, api_key: str | None) -> Client:
        return Client(base_url=base_url, api_key=api_key, transport=httpx.MockTransport(respond))

    monkeypatch.setenv("KAYAK_BASE_URL", "http://fixture.test")
    monkeypatch.setenv("KAYAK_API_KEY", "fixture-token")
    monkeypatch.setattr(rag_decisions, "retrieve", retrieve)
    monkeypatch.setattr(rag_decisions, "Client", client)
    rag_decisions.main()

    assert queries == ["How long are audit logs kept?"]
    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert str(calls[0].url) == "http://fixture.test/v1/decide"
    assert calls[0].headers["authorization"] == "Bearer fixture-token"
    request = DecisionRequest.model_validate_json(calls[0].content)
    assert json.loads(request.state) == {"query": queries[0], "context": documents}
    assert set(request.questions) == {"retention", "stated", "completeness"}
    assert request.questions["retention"].criteria == {
        "30_days": "30 days",
        "90_days": "90 days",
        "unknown": "Not stated",
    }
    assert request.questions["stated"].criteria == {
        "false": "false: No. This is false: The context states the audit-log retention period.",
        "true": "true: Yes. This is true: The context states the audit-log retention period.",
    }
    assert request.questions["completeness"].criteria == {
        "0": "No answer",
        "1": "Partial answer",
        "2": "Complete answer",
    }

    result = JudgmentResult.model_validate_json(capsys.readouterr().out)
    retention = result.answers["retention"]
    assert isinstance(retention, ChoiceAnswer) and retention.choice == selected
    assert retention == wire_result.answers["retention"]
    stated = result.answers["stated"]
    assert isinstance(stated, NoulAnswer)
    assert stated.noul == pytest.approx(math.exp(2) / (1 + math.exp(2)))
    assert stated.scores == wire_result.answers["stated"].scores
    assert stated.probabilities == wire_result.answers["stated"].probabilities
    completeness = result.answers["completeness"]
    assert isinstance(completeness, ScoreAnswer)
    assert completeness.score == pytest.approx(
        (math.exp(1) + 2 * math.exp(3)) / (1 + math.exp(1) + math.exp(3))
    )
    assert completeness.legend == request.questions["completeness"].criteria
    assert completeness.scores == wire_result.answers["completeness"].scores
    assert completeness.probabilities == wire_result.answers["completeness"].probabilities
    assert result.model == wire_result.model
    assert result.input_tokens == 31 and result.calibration == "none"


def test_empty_retrieval_stops_before_client_construction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    constructions: list[str] = []

    def unexpected_client(*, base_url: str, api_key: str | None) -> Client:
        constructions.append(base_url)
        raise AssertionError("Empty retrieval must not construct a client")

    monkeypatch.setattr(rag_decisions, "retrieve", lambda query: {})
    monkeypatch.setattr(rag_decisions, "Client", unexpected_client)
    with pytest.raises(ValueError, match="No evidence retrieved"):
        rag_decisions.main()
    assert constructions == []
    assert capsys.readouterr().out == ""


def test_provider_error_propagates_without_success_output_or_retry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            503, json={"error": {"code": "fixture_unavailable", "message": "Service unavailable"}}
        )

    def client(*, base_url: str, api_key: str | None) -> Client:
        return Client(base_url=base_url, api_key=api_key, transport=httpx.MockTransport(respond))

    monkeypatch.setenv("KAYAK_BASE_URL", "http://fixture.test")
    monkeypatch.delenv("KAYAK_API_KEY", raising=False)
    monkeypatch.setattr(rag_decisions, "Client", client)
    with pytest.raises(RemoteError) as error:
        rag_decisions.main()
    assert error.value.status_code == 503
    assert error.value.code == "fixture_unavailable"
    assert len(calls) == 1
    assert calls[0].url.path == "/v1/decide"
    assert capsys.readouterr().out == ""


def test_complete_example_is_under_fifty_physical_lines() -> None:
    assert len(Path(rag_decisions.__file__).read_text(encoding="utf-8").splitlines()) < 50
