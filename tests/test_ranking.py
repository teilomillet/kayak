"""Observe ranking through Python and CLI without changing the Choice recipe."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_cli import run_cli
from test_contract import sample_result

from kayak import (
    Choice,
    Client,
    DecisionRequest,
    InputError,
    Model,
    ModelClosedError,
    RankingRequest,
    RankingResult,
    TransportError,
)
from kayak.decisions import MAX_CANDIDATES, MAX_REQUEST_BYTES, answer_from_scores
from kayak.runtime._preparation import prepare_texts


def test_ranking_uses_v1_and_keeps_ties_and_full_distribution() -> None:
    candidates = {"third": "same text", "first": "same text", "second": "other text"}
    calls: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        assert wire.url.path == "/v1/decide"
        request = DecisionRequest.model_validate_json(wire.content)
        assert list(request.questions) == ["rank"]
        assert request.questions["rank"].criteria == candidates
        result = sample_result().model_copy(
            update={"answers": {"rank": answer_from_scores(list(candidates), [-1.0, -1.0, 2.0])}}
        )
        candidates.clear()  # Caller mutation after serialization cannot change the result.
        return httpx.Response(200, content=result.model_dump_json())

    with Client(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as client:
        result = client.rank(state="context", instructions="Pick one", candidates=candidates)
    assert len(calls) == 1
    assert [candidate.id for candidate in result.ranked] == ["second", "third", "first"]
    assert [candidate.score for candidate in result.ranked] == [2.0, -1.0, -1.0]
    assert result.ranked[1].probability == result.ranked[2].probability
    assert 0.9 < result.ranked[0].probability < 1.0
    assert sum(candidate.probability for candidate in result.ranked) == pytest.approx(1.0)
    assert sum(candidate.probability for candidate in result.ranked[:2]) < 1.0
    assert RankingResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "payload",
    [
        {"state": " "},
        {"instructions": " "},
        {"candidates": {}},
        {"candidates": {"a": " "}},
        {"candidates": {"a": None}},
        {"candidates": {" ": "text"}},
        {"candidates": ["first", "second"]},
        {"candidates": {str(i): "text" for i in range(MAX_CANDIDATES + 1)}},
        {"candidates": {str(i): "x" * 65_536 for i in range(4)}},
        {"extra": "unknown"},
    ],
)
def test_invalid_ranking_inputs_share_choice_limits(payload: dict[str, object]) -> None:
    body = {"state": "text", "instructions": "Choose", "candidates": {"a": "text"}, **payload}
    with pytest.raises(ValidationError):
        RankingRequest.model_validate(body)


def test_ranking_snapshot_and_recipe_preserve_unicode_whitespace_and_ids() -> None:
    candidates = {"réponse": "  Oui.\n", "different-id": "  Oui.\n"}
    request = RankingRequest(state="  état \n", instructions="  Question? ", candidates=candidates)
    candidates.clear()
    decision = request.as_decision()
    assert prepare_texts(decision) == ["état\n\nQuestion?", "  Oui.\n", "  Oui.\n"]
    request.candidates.clear()
    assert list(decision.questions["rank"].criteria) == ["réponse", "different-id"]
    with pytest.raises(ValidationError):
        request.as_decision()


def test_rank_rejects_bad_inputs_before_http_and_checks_the_response() -> None:
    calls: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        return httpx.Response(200, content=sample_result().model_dump_json())

    with Client(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(InputError):
            client.rank(state="text", instructions="Choose", candidates={})
        assert calls == []
        with pytest.raises(TransportError):
            client.rank(state="text", instructions="Choose", candidates={"a": "text"})
        assert len(calls) == 1


@pytest.mark.inference
def test_rank_and_decide_have_identical_real_tiny_model_outputs(tiny_model: Model) -> None:
    candidates = {"billing": "billing refunds", "support": "technical service outages"}
    direct = tiny_model.decide(
        state="charged twice",
        questions={"chosen": Choice(instructions="Which team?", criteria=candidates)},
    )
    ranked = tiny_model.rank(
        state="charged twice", instructions="Which team?", candidates=candidates
    )
    answer = direct.answers["chosen"]
    assert ranked.ranked[0].id == answer.choice
    assert {item.id: item.score for item in ranked.ranked} == answer.scores
    assert {item.id: item.probability for item in ranked.ranked} == answer.probabilities
    assert ranked.input_tokens == direct.input_tokens
    assert ranked.model == direct.model
    with pytest.raises(InputError, match="tokens"):
        tiny_model.rank(state="charged " * 2048, instructions="Which team?", candidates=candidates)
    assert (
        tiny_model.rank(state="charged twice", instructions="Which team?", candidates=candidates)
        == ranked
    )
    tiny_model.close()
    with pytest.raises(ModelClosedError):
        tiny_model.rank(state="charged twice", instructions="Which team?", candidates=candidates)


def test_cli_ranking_validation_and_stdin() -> None:
    source = Path("examples/ranking.json").read_text()
    for arguments in (
        ("validate", "--ranking", "-"),
        ("validate", "--ranking", "examples/ranking.json", "--pretty"),
    ):
        completed = run_cli(*arguments, input_text=source)
        assert completed.returncode == 0, completed.stderr
        assert not completed.stderr
        assert RankingRequest.model_validate_json(
            completed.stdout
        ) == RankingRequest.model_validate_json(source)
    help_output = run_cli("rank", "--help")
    assert help_output.returncode == 0
    assert "--base-url" in help_output.stdout


@pytest.mark.parametrize("payload", ["{", '{"state":"private","candidates":{}}'])
def test_cli_bad_rank_fails_before_connection(payload: str) -> None:
    completed = run_cli("rank", "-", "--base-url", "http://127.0.0.1:1", input_text=payload)
    assert completed.returncode == 2
    assert not completed.stdout
    assert "private" not in completed.stderr
    assert "Traceback" not in completed.stderr


def test_cli_rank_input_bytes_are_bounded(tmp_path: Path) -> None:
    raw = Path("examples/ranking.json").read_bytes()
    path = tmp_path / "rank.json"
    path.write_bytes(raw + b" " * (MAX_REQUEST_BYTES - len(raw)))
    completed = run_cli("validate", "--ranking", str(path))
    assert completed.returncode == 0
    assert json.loads(completed.stdout) == json.loads(raw)
    path.write_bytes(path.read_bytes() + b" ")
    completed = run_cli("rank", str(path))
    assert completed.returncode == 2
    assert "exceeds 1 MiB" in completed.stderr
    assert not completed.stdout
