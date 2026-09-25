from __future__ import annotations

import json
import subprocess
import sys

import httpx
import pytest
from pydantic import ValidationError

from kayak import Choice, Client, InputError, RemoteError, TransportError
from kayak.decisions import (
    DecisionRequest,
    DecisionResult,
    ModelInfo,
    answer_from_scores,
    request_from,
    validation_message,
)


def sample_result() -> DecisionResult:
    return DecisionResult(
        model=ModelInfo(
            id="fixture",
            revision="1",
            fingerprint="f",
            encoder="fixture",
            encoder_revision="a" * 40,
            device="cpu",
            dtype="float32",
        ),
        answers={"route": answer_from_scores(["billing", "support"], [5.0, 0.0])},
        input_tokens=4,
    )


def test_import_does_not_load_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import kayak, sys; "
            "assert not hasattr(kayak, 'rag') and 'rag' not in kayak.__all__; "
            "from kayak.decisions import Choice, ChoiceAnswer, DecisionResult, ModelInfo; "
            "from kayak.runtime import Model, load; "
            "assert (Choice, ChoiceAnswer, DecisionResult, ModelInfo) == "
            "(kayak.Choice, kayak.ChoiceAnswer, kayak.DecisionResult, kayak.ModelInfo); "
            "assert Model is kayak.Model and load is kayak.load; "
            "assert not {'torch', 'transformers', 'fastapi', 'huggingface_hub'} & "
            "sys.modules.keys()",
        ],
        check=True,
    )


def test_selection_distribution_and_ties() -> None:
    answer = answer_from_scores(["a", "b"], [5.0, 0.0])
    assert answer.choice == "a"
    assert answer.probabilities["a"] == pytest.approx(0.9933071490757153)
    assert answer_from_scores(["b", "a"], [0.0, 0.0]).choice == "b"


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], []])
def test_invalid_scores_are_errors(scores: list[float]) -> None:
    from kayak import InferenceError

    with pytest.raises(InferenceError):
        answer_from_scores([str(i) for i in range(len(scores))], scores)


def test_typed_request_snapshots_mutable_input(questions: dict[str, Choice]) -> None:
    request = request_from("charged twice", questions)
    questions["route"].criteria["billing"] = "changed"
    assert request.questions["route"].criteria["billing"] == "billing refunds"


@pytest.mark.parametrize(
    "patch",
    [
        {"state": {}},
        {"state": "  "},
        {"questions": {}},
        {"questions": {"x": {"type": "noul", "instructions": "true?"}}},
        {"questions": {"x": {"type": "choice", "instructions": "pick", "criteria": {"x": ""}}}},
        {"extra": True},
    ],
)
def test_request_rejects_unsupported_meanings(
    questions: dict[str, Choice], patch: dict[str, object]
) -> None:
    payload = request_from("charged twice", questions).model_dump()
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({**payload, **patch})


def test_total_candidates_limited() -> None:
    q = Choice(instructions="pick", criteria={str(i): "candidate" for i in range(129)})
    with pytest.raises(InputError, match="256 candidates"):
        request_from("state", {"a": q, "b": q})


@pytest.mark.parametrize(
    "question",
    [
        {"instructions": 42, "criteria": {"candidate": "description"}},
        {"instructions": "choose", "criteria": {42: "description"}},
    ],
)
def test_migration_hints_do_not_mislabel_instruction_or_candidate_id_errors(
    question: dict[str, object],
) -> None:
    with pytest.raises(ValidationError) as exc:
        DecisionRequest.model_validate({"state": "state", "questions": {"criteria": question}})
    assert "explicit text description" not in validation_message(exc.value)


def test_client_round_trip(questions: dict[str, Choice]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/decide"
        assert request.headers["authorization"] == "Bearer token"
        assert json.loads(request.content)["questions"]["route"]["type"] == "choice"
        return httpx.Response(200, json=sample_result().model_dump())

    with Client(
        base_url="http://local", api_key="token", transport=httpx.MockTransport(respond)
    ) as c:
        assert c.decide(state="charged twice", questions=questions) == sample_result()


@pytest.mark.parametrize("change", ["ids", "nan", "distribution", "choice"])
def test_client_rejects_invalid_results(questions: dict[str, Choice], change: str) -> None:
    payload = sample_result().model_dump()
    answer = payload["answers"]["route"]
    if change == "ids":
        payload["answers"]["wrong"] = payload["answers"].pop("route")
    elif change == "nan":
        answer["scores"]["billing"] = "NaN"
    elif change == "distribution":
        answer["probabilities"] = {"billing": 0.5, "support": 0.5}
    else:
        answer["choice"] = "support"
    with Client(
        base_url="http://local",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    ) as c:
        with pytest.raises(TransportError):
            c.decide(state="charged twice", questions=questions)


def test_client_does_not_retry_or_follow_redirects(questions: dict[str, Choice]) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(307, headers={"location": "http://elsewhere/v1/decide"})

    with Client(base_url="http://local", transport=httpx.MockTransport(respond)) as c:
        with pytest.raises(RemoteError) as error:
            c.decide(state="charged twice", questions=questions)
    assert error.value.status_code == 307
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"<html>unavailable</html>", b'{"error":', b"\xff"])
def test_client_error_fallback_for_unreadable_json(
    questions: dict[str, Choice], body: bytes
) -> None:
    with Client(
        base_url="http://local",
        transport=httpx.MockTransport(lambda _: httpx.Response(503, content=body)),
    ) as client:
        with pytest.raises(RemoteError) as error:
            client.decide(state="charged twice", questions=questions)
    assert error.value.status_code == 503
    assert error.value.code == "http_error"
    assert str(error.value) == "Kayak server returned HTTP 503"
