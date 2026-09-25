"""Generated checks of public contracts, independent of model computation."""

from __future__ import annotations

import json

import httpx
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import JsonValue, ValidationError
from starlette.responses import JSONResponse, Response

from kayak import Choice, Client, InputError, RemoteError, TransportError
from kayak.decisions import (
    MAX_CANDIDATES,
    MAX_QUESTIONS,
    MAX_REQUEST_CHARS,
    DecisionRequest,
    DecisionResult,
    ModelInfo,
    answer_from_scores,
    request_from,
)

nonblank = st.text(min_size=1, max_size=24).filter(lambda text: bool(text.strip()))
choices = st.builds(
    Choice,
    instructions=nonblank,
    criteria=st.dictionaries(nonblank, nonblank, min_size=1, max_size=8),
)
questions = st.dictionaries(nonblank, choices, min_size=1, max_size=4)
info = ModelInfo(
    id="fixture",
    revision="1",
    fingerprint="f",
    encoder="fixture",
    encoder_revision="a" * 40,
    device="cpu",
    dtype="float32",
)


@given(state=nonblank, typed=questions)
def test_request_roundtrip_preserves_text_order_and_ownership(
    state: str, typed: dict[str, Choice]
) -> None:
    raw = {key: question.model_dump() for key, question in typed.items()}
    from_typed = request_from(state, typed)
    from_raw = request_from(state, raw)
    encoded = from_typed.model_dump_json().encode()
    assert from_typed == from_raw == DecisionRequest.model_validate_json(bytearray(encoded))
    assert from_typed.state == state
    assert list(from_typed.questions) == list(typed)
    for key, question in typed.items():
        assert list(from_typed.questions[key].criteria) == list(question.criteria)
        assert from_typed.questions[key] is not question
        assert from_typed.questions[key].criteria is not question.criteria
        question.criteria.clear()
        raw[key]["criteria"].clear()
    assert from_typed == from_raw == DecisionRequest.model_validate_json(encoded)


@given(question=choices)
def test_mutated_typed_input_is_revalidated(question: Choice) -> None:
    question.criteria[next(iter(question.criteria))] = " "
    with pytest.raises(InputError):
        request_from("state", {"question": question})


@given(state=nonblank, typed=questions)
def test_client_and_json_encoders_preserve_decisions(state: str, typed: dict[str, Choice]) -> None:
    request = request_from(state, typed)
    result = DecisionResult(
        model=info,
        answers={
            key: answer_from_scores(list(q.criteria), [float(i) for i in range(len(q.criteria))])
            for key, q in typed.items()
        },
        input_tokens=0,
    )
    old = JSONResponse(result.model_dump())
    direct = Response(result.model_dump_json(), media_type="application/json")
    assert old.headers["content-type"] == direct.headers["content-type"]
    assert json.loads(bytes(old.body)) == json.loads(bytes(direct.body))
    assert DecisionResult.model_validate_json(bytes(direct.body)) == result
    calls = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        assert wire.headers["content-type"] == "application/json"
        assert wire.content == request.model_dump_json().encode()
        assert int(wire.headers["content-length"]) == len(wire.content)
        assert DecisionRequest.model_validate_json(wire.content) == request
        return httpx.Response(200, content=bytes(direct.body))

    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        assert client.decide(state=state, questions=typed) == result
    assert len(calls) == 1


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5)
    ),
    max_leaves=20,
)


@given(
    payload=json_values
    | st.fixed_dictionaries(
        {"error": st.fixed_dictionaries({"code": nonblank, "message": st.text()})}
    ),
    status=st.sampled_from([301, 400, 401, 413, 422, 500, 503, 504]),
)
@example(payload={"error": {"code": "invalid_request", "message": "bad input"}}, status=422)
@example(payload={"error": []}, status=500)
@example(payload={"error": {"message": "missing code"}}, status=500)
@example(payload={"error": {"code": "invalid_request", "message": 42}}, status=422)
def test_untrusted_error_bodies_have_bounded_failure_semantics(
    payload: JsonValue, status: int
) -> None:
    calls = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        return httpx.Response(status, json=payload)

    error = payload.get("error") if isinstance(payload, dict) else None
    if (
        isinstance(error, dict)
        and isinstance(error.get("code"), str)
        and isinstance(error.get("message"), str)
    ):
        code, message = error["code"], error["message"]
    else:
        code, message = "http_error", None
    expected = InputError if status == 422 and code == "invalid_request" else RemoteError
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(expected) as failure:
            client.decide(
                state="state", questions={"q": Choice(instructions="pick", criteria={"a": "a"})}
            )
    if isinstance(failure.value, RemoteError):
        assert failure.value.status_code == status
        assert failure.value.code == code
    if message is not None:
        assert str(failure.value) == message
    else:
        assert str(failure.value) == f"Kayak server returned HTTP {status}"
    assert len(calls) == 1


@given(body=st.binary(max_size=512))
def test_untrusted_success_body_returns_a_decision_or_transport_error(body: bytes) -> None:
    with Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
    ) as client:
        try:
            result = client.decide(
                state="state", questions={"q": Choice(instructions="pick", criteria={"a": "a"})}
            )
        except TransportError:
            return
        assert isinstance(result, DecisionResult)
        assert set(result.answers) == {"q"}
        assert list(result.answers["q"].scores) == ["a"]


@given(count=st.integers(min_value=MAX_CANDIDATES - 2, max_value=MAX_CANDIDATES + 2))
def test_total_candidate_boundary(count: int) -> None:
    # Spread candidates across two individually valid questions.
    questions = {
        str(i): Choice(instructions="pick", criteria={str(j): "option" for j in range(n)})
        for i, n in enumerate((count // 2, count - count // 2))
    }
    if count <= MAX_CANDIDATES:
        assert (
            sum(len(q.criteria) for q in request_from("s", questions).questions.values()) == count
        )
    else:
        with pytest.raises(InputError, match="candidates"):
            request_from("s", questions)


@given(count=st.integers(min_value=MAX_QUESTIONS - 2, max_value=MAX_QUESTIONS + 2))
def test_question_boundary(count: int) -> None:
    question = Choice(instructions="pick", criteria={"a": "option"})
    questions = {str(i): question for i in range(count)}
    if count <= MAX_QUESTIONS:
        assert len(request_from("s", questions).questions) == count
    else:
        with pytest.raises(InputError):
            request_from("s", questions)


@given(delta=st.integers(min_value=-2, max_value=2))
def test_aggregate_character_boundary(delta: int) -> None:
    # Include IDs, instructions, and descriptions in the independent size calculation.
    texts = ["x" * 65_536] * 3 + ["x" * (65_536 - 7 + delta)]
    question = Choice(instructions="i", criteria=dict(zip("abcd", texts, strict=True)))
    total = len("s") + len("q") + len("i") + sum(1 + len(t) for t in texts)
    assert total == MAX_REQUEST_CHARS + delta
    if delta <= 0:
        request_from("s", {"q": question})
    else:
        with pytest.raises(InputError, match="characters"):
            request_from("s", {"q": question})


@given(extra=nonblank)
def test_unknown_fields_are_rejected(extra: str) -> None:
    if extra not in {"state", "questions"}:
        with pytest.raises(ValidationError):
            DecisionRequest.model_validate(
                {
                    "state": "s",
                    "questions": {"q": Choice(instructions="i", criteria={"a": "a"})},
                    extra: True,
                }
            )


@pytest.mark.parametrize(
    "location", ["state", "instructions", "question_id", "candidate_id", "description"]
)
@given(text=st.text(max_size=24))
@example(text="")
@example(text=" \t\x1c\x1d\x1e\x1f\u0085\u00a0\u1680\u2007\u2028\u2029\u202f\u205f\u3000 ")
@example(text="\u200b")
@example(text="  unchanged text 🚣  ")
def test_blank_checks_preserve_python_whitespace_rules_and_original_text(
    location: str, text: str
) -> None:
    state = text if location == "state" else "state"
    question_id = text if location == "question_id" else "question"
    candidate_id = text if location == "candidate_id" else "candidate"
    instructions = text if location == "instructions" else "choose"
    description = text if location == "description" else "description"
    questions = {
        question_id: {"instructions": instructions, "criteria": {candidate_id: description}}
    }
    if not text.strip():
        with pytest.raises(InputError):
            request_from(state, questions)
        return
    request = request_from(state, questions)
    assert request.state == state
    assert list(request.questions) == [question_id]
    assert request.questions[question_id].instructions == instructions
    assert request.questions[question_id].criteria == {candidate_id: description}
