"""Typed adapters preserve the pinned recipe, explicit uncertainty, and /v1 behavior."""

import asyncio
import json
import math
import traceback
from collections.abc import Mapping
from typing import assert_type

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from test_contract import sample_result

from kayak import (
    AsyncClient,
    Choice,
    Client,
    DecisionRequest,
    DecisionResult,
    InferenceError,
    InputError,
    JudgmentQuestion,
    JudgmentRequest,
    JudgmentResult,
    Model,
    ModelClosedError,
    Noul,
    NoulAnswer,
    RemoteError,
    Score,
    ScoreAnswer,
    TransportError,
)
from kayak.decisions import answer_from_scores
from kayak.runtime._preparation import prepare_texts


def mixed_request() -> JudgmentRequest:
    return JudgmentRequest(
        state="  charged twice\n",
        questions={
            "route": Choice(instructions="Which team?", criteria={"a": "billing", "b": "support"}),
            "urgent": Noul(instructions="  Is this urgent?\n"),
            "severity": Score(instructions="Assess impact", criteria=["Low", "Medium", "High"]),
        },
    )


def response_for(request: DecisionRequest) -> DecisionResult:
    return DecisionResult(
        model=sample_result().model,
        input_tokens=31,
        answers={
            name: answer_from_scores(list(question.criteria), [0.0] * len(question.criteria))
            for name, question in request.questions.items()
        },
    )


def test_pinned_text_recipe_and_ties_have_explicit_meaning() -> None:
    request = mixed_request()
    decision = request.as_decision()
    assert prepare_texts(decision) == [
        "charged twice\n\nWhich team?",
        "charged twice\n\nIs this urgent?",
        "charged twice\n\nAssess impact",
        "billing",
        "support",
        "false: No. This is false: Is this urgent?",
        "true: Yes. This is true: Is this urgent?",
        "Low",
        "Medium",
        "High",
    ]
    result = request.decode(response_for(decision))
    assert_type(result, JudgmentResult)
    noul, score, choice = (result.answers[key] for key in ("urgent", "severity", "route"))
    assert isinstance(noul, NoulAnswer) and noul.noul == 0.5
    assert noul.scores == {"false": 0.0, "true": 0.0}
    assert noul.probabilities == {"false": 0.5, "true": 0.5}
    assert not hasattr(noul, "choice")  # No automatic true/false policy, especially on ties.
    assert isinstance(score, ScoreAnswer) and score.score == 1.0
    assert score.legend == {"0": "Low", "1": "Medium", "2": "High"}
    assert score.probabilities == dict.fromkeys(("0", "1", "2"), 1 / 3)
    assert choice.type == "choice" and choice.choice == "a"
    assert result.input_tokens == 31 and result.calibration == "none"
    assert JudgmentResult.model_validate_json(result.model_dump_json()) == result
    assert JudgmentRequest.model_validate_json(request.model_dump_json()) == request


def test_custom_noul_defaults_and_rubric_text_are_preserved() -> None:
    request = JudgmentRequest(
        state="café",
        questions={
            "n": Noul(instructions="Vrai?", criteria={"true": "  Oui.\n", "false": ""}),
            "s": Score(instructions="Level?", criteria=["  Bas\n", "Élevé"]),
        },
    ).as_decision()
    assert request.questions["n"].criteria == {
        "false": "false: No. This is false: Vrai?",
        "true": "true:   Oui.\n",
    }
    assert request.questions["s"].criteria == {"0": "  Bas\n", "1": "Élevé"}


@given(st.lists(st.floats(min_value=-50, max_value=50), min_size=2, max_size=16))
def test_expected_index_uses_ids_not_probability_mapping_order(logits: list[float]) -> None:
    request = JudgmentRequest(
        state="context",
        questions={"s": Score(instructions="Assess", criteria=["text"] * len(logits))},
    )
    keys = list(request.as_decision().questions["s"].criteria)
    answer = answer_from_scores(keys, logits)
    answer = answer.model_copy(
        update={"probabilities": dict(reversed(list(answer.probabilities.items())))}
    )
    result = request.decode(
        DecisionResult(model=sample_result().model, input_tokens=0, answers={"s": answer})
    )
    scored = result.answers["s"]
    assert isinstance(scored, ScoreAnswer)
    # Independent arithmetic: weighted softmax numerator divided by its denominator.
    weights = [math.exp(logit - max(logits)) for logit in logits]
    expected = sum(index * weight for index, weight in enumerate(weights)) / sum(weights)
    assert scored.score == pytest.approx(expected, abs=1e-12)
    assert 0 <= scored.score <= len(logits) - 1


def test_middle_score_does_not_hide_bimodal_distribution() -> None:
    request = JudgmentRequest(
        state="context",
        questions={"s": Score(instructions="Assess", criteria=["Low", "Mid", "High"])},
    )
    answer = answer_from_scores(["0", "1", "2"], [0.0, -1000.0, 0.0])
    result = request.decode(
        DecisionResult(model=sample_result().model, input_tokens=0, answers={"s": answer})
    )
    score = result.answers["s"]
    assert isinstance(score, ScoreAnswer)
    assert score.score == 1 and score.probabilities == {"0": 0.5, "1": 0.0, "2": 0.5}


@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul", "instructions": " "},
        {"type": "noul", "instructions": "True?", "criteria": {"other": "x"}},
        {"type": "noul", "instructions": "True?", "criteria": {"true": None}},
        {"type": "noul", "instructions": "True?", "criteria": {"true": 1}},
        {"type": "noul", "instructions": "True?", "criteria": {"true": "x" * 65_536}},
        {"type": "score", "instructions": "Assess", "criteria": ["only one"]},
        {"type": "score", "instructions": "Assess", "criteria": ["low", " "]},
        {"type": "score", "instructions": "Assess", "criteria": {"0": "low", "1": "high"}},
        {"type": "score", "instructions": "Assess", "criteria": ["level"] * 257},
        {"type": "score", "instructions": "Assess", "criteria": ["low", "high"], "threshold": 0.8},
        {"type": "invented", "instructions": "Assess"},
        {"type": "PRIVATE_CUSTOMER_CONTENT", "instructions": "Assess"},
    ],
)
def test_invalid_questions_fail_before_network_without_input_in_traceback(
    question: dict[str, object],
) -> None:
    private_state = "PRIVATE_CUSTOMER_CONTENT"
    with Client(
        base_url="http://fixture.test",
        transport=httpx.MockTransport(lambda _: pytest.fail("network request")),
    ) as client:
        with pytest.raises(InputError) as error:
            client.judge(state=private_state, questions={"q": question})
    assert "PRIVATE_CUSTOMER_CONTENT" not in "".join(traceback.format_exception(error.value))


def test_aggregate_limits_and_mutated_typed_inputs_are_revalidated() -> None:
    with pytest.raises(ValidationError, match="256 candidates"):
        JudgmentRequest(
            state="context",
            questions={
                "a": Score(instructions="Assess", criteria=["level"] * 255),
                "b": Noul(instructions="True?"),
            },
        )
    with pytest.raises(ValidationError):
        JudgmentRequest(
            state="context", questions={str(i): Noul(instructions="True?") for i in range(33)}
        )
    request = mixed_request()
    score = request.questions["severity"]
    assert isinstance(score, Score)
    score.criteria[:] = ["only one"]
    with pytest.raises(ValidationError):
        request.as_decision()
    with Client(
        base_url="http://fixture.test",
        transport=httpx.MockTransport(lambda _: pytest.fail("network request")),
    ) as client:
        with pytest.raises(InputError):
            client.judge(state=request.state, questions=request.questions)


@pytest.mark.parametrize("dictionary", [False, True])
def test_sync_and_async_use_one_choice_call_and_snapshot_originals(dictionary: bool) -> None:
    request = mixed_request()
    original = request.as_decision()
    questions: Mapping[str, JudgmentQuestion | Mapping[str, object]] = request.questions
    if dictionary:
        questions = {name: question.model_dump() for name, question in request.questions.items()}
    calls: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        assert wire.url.path == "/prefix/v1/decide"
        assert wire.headers["authorization"] == "Bearer fixture"
        assert set(json.loads(wire.content)) == {"state", "questions"}
        parsed = DecisionRequest.model_validate_json(wire.content)
        assert parsed == original
        if not dictionary:
            score = request.questions["severity"]
            assert isinstance(score, Score)
            score.criteria.clear()  # The decoding recipe must already be an owned snapshot.
        return httpx.Response(200, content=response_for(parsed).model_dump_json())

    expected = request.decode(response_for(original))
    with Client(
        base_url="http://fixture.test/prefix",
        api_key="fixture",
        transport=httpx.MockTransport(respond),
    ) as client:
        assert client.judge(state=request.state, questions=questions) == expected
    request = mixed_request()
    questions = (
        request.questions
        if not dictionary
        else {name: question.model_dump() for name, question in request.questions.items()}
    )

    async def scenario() -> None:
        async with AsyncClient(
            base_url="http://fixture.test/prefix",
            api_key="fixture",
            transport=httpx.MockTransport(respond),
        ) as client:
            assert await client.judge(state=request.state, questions=questions) == expected

    asyncio.run(scenario())
    assert len(calls) == 2


@pytest.mark.parametrize("status", [503, 504, 307])
def test_judge_preserves_errors_without_retrying(status: int) -> None:
    calls = 0

    def respond(wire: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"code": "fixture", "message": "failure"}})

    with Client(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(RemoteError) as error:
            client.judge(state="context", questions={"q": Noul(instructions="True?")})
    assert error.value.status_code == status and calls == 1


def test_decode_rejects_unrelated_or_inconsistent_results() -> None:
    request = mixed_request()
    with pytest.raises(InferenceError):
        request.decode(sample_result())
    result = request.decode(response_for(request.as_decision()))
    payload = result.model_dump()
    payload["answers"]["urgent"]["noul"] = 0.9
    with pytest.raises(ValidationError, match="true candidate"):
        JudgmentResult.model_validate(payload)
    payload = result.model_dump()
    payload["answers"]["severity"]["score"] = 2.0
    with pytest.raises(ValidationError, match="probability-weighted"):
        JudgmentResult.model_validate(payload)
    with Client(
        base_url="http://fixture.test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=sample_result().model_dump_json())
        ),
    ) as client:
        with pytest.raises(TransportError):
            client.judge(state=request.state, questions=request.questions)


@pytest.mark.inference
def test_judge_matches_tiny_model_and_http_without_changing_execution(tiny_model: Model) -> None:
    from fastapi.testclient import TestClient

    from kayak.server import create_app

    request = mixed_request()
    decision = request.as_decision()
    raw = tiny_model.decide(state=decision.state, questions=decision.questions)
    expected = request.decode(raw)
    actual = tiny_model.judge(state=request.state, questions=request.questions)
    assert actual == expected
    with TestClient(create_app(lambda: tiny_model)) as service:
        # Send the compiled request through the real service validation and model.
        wire = service.post("/v1/decide", json=decision.model_dump())
        assert wire.status_code == 200
        assert request.decode(DecisionResult.model_validate_json(wire.content)) == expected
        assert service.post("/v1/decide", json=request.model_dump()).status_code == 422

        def forward(wire: httpx.Request) -> httpx.Response:
            response: object = service.post(
                wire.url.path, content=wire.content, headers=wire.headers
            )
            assert isinstance(response, httpx.Response)
            return response

        with Client(
            base_url="http://fixture.test", transport=httpx.MockTransport(forward)
        ) as client:
            assert client.judge(state=request.state, questions=request.questions) == expected
    with pytest.raises(ModelClosedError):
        tiny_model.judge(state=request.state, questions=request.questions)


@pytest.mark.inference
def test_judge_token_overflow_fails_before_forward(
    tiny_model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiny_model._encoder, "forward", lambda **kw: pytest.fail("ran inference"))
    with pytest.raises(InputError, match="never silently truncated"):
        tiny_model.judge(state="charged " * 2048, questions={"q": Noul(instructions="True?")})


def test_standalone_example_uses_public_client_and_labels_its_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from examples import typed_judgments

    calls: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        assert wire.headers["authorization"] == "Bearer example-key"
        request = DecisionRequest.model_validate_json(wire.content)
        return httpx.Response(200, content=response_for(request).model_dump_json())

    def client(*, base_url: str, api_key: str | None) -> Client:
        return Client(base_url=base_url, api_key=api_key, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(typed_judgments, "Client", client)
    monkeypatch.setenv("KAYAK_BASE_URL", "http://fixture.test")
    monkeypatch.setenv("KAYAK_API_KEY", "example-key")
    typed_judgments.main()
    output = capsys.readouterr().out
    assert "True-candidate share (uncalibrated): 0.500" in output
    assert "Expected impact level (0–2): 1.000" in output
    assert '"calibration": "none"' in output and '"legend":' in output
    assert len(calls) == 1
