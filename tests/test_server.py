from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from kayak import Choice, Client, InferenceError, InputError, Model, ModelClosedError
from kayak.decisions import DecisionResult, ModelInfo, answer_from_scores
from kayak.server import MAX_BODY_BYTES, create_app


class ControlledModel:
    """A controlled effect for admission/lifetime tests; not model quality evidence."""

    info = ModelInfo(
        id="controlled",
        revision="1",
        fingerprint="f",
        encoder="controlled",
        encoder_revision="a" * 40,
        device="cpu",
        dtype="float32",
    )

    def __init__(self) -> None:
        self.started, self.release = Event(), Event()
        self.closed = False

    def decide(self, *, state: str, questions: Mapping[str, Choice]) -> DecisionResult:
        if state == "block":
            self.started.set()
            assert self.release.wait(5), "test did not release the inference operation"
        return DecisionResult(
            model=self.info,
            input_tokens=1,
            answers={
                k: answer_from_scores(list(q.criteria), [0.0] * len(q.criteria))
                for k, q in questions.items()
            },
        )

    def close(self) -> None:
        self.closed = True


@pytest.mark.inference
def test_http_matches_direct_with_actual_tiny_model(
    tiny_model: Model, questions: Mapping[str, Choice]
) -> None:
    direct = tiny_model.decide(state="charged twice", questions=questions)
    app = create_app(lambda: tiny_model, api_key="secret")
    with TestClient(app) as http:

        def inspect(request: httpx.Request) -> httpx.Response:
            assert request.method == "GET" and request.url.path == "/v1/model"
            response: object = http.get(request.url.path, headers=request.headers)
            assert isinstance(response, httpx.Response)
            return response

        with Client(
            base_url="http://test", api_key="secret", transport=httpx.MockTransport(inspect)
        ) as client:
            assert client.model_info() == tiny_model.info
        assert http.get("/health").json() == {"ready": True}
        assert http.post("/v1/decide", json={}).status_code == 401
        response = http.post(
            "/v1/decide",
            headers={"authorization": "Bearer secret"},
            json={
                "state": "charged twice",
                "questions": {k: q.model_dump() for k, q in questions.items()},
            },
        )
        assert response.status_code == 200
        assert DecisionResult.model_validate(response.json()) == direct
    assert tiny_model._encoder is None


@pytest.mark.inference
def test_portable_validator_over_live_http(tiny_model: Model) -> None:
    from scripts.validate_model import validate_http

    case = ("charged twice", "Which team?", {"billing": "billing", "support": "technical"})
    state, instructions, criteria = case
    direct = tiny_model.decide(
        state=state, questions={"decision": Choice(instructions=instructions, criteria=criteria)}
    )
    assert validate_http(tiny_model, case, direct)
    assert tiny_model._encoder is None


def test_http_rejects_invalid_and_oversized_input() -> None:
    model = ControlledModel()
    with TestClient(create_app(lambda: model)) as http:
        metadata = http.get("/v1/model")
        assert metadata.status_code == 200
        assert ModelInfo.model_validate_json(metadata.content) == model.info
        response = http.post("/v1/decide", content='{"state": {"private": "secret"}}')
        assert response.status_code == 422
        assert "secret" not in response.text
        assert http.post("/v1/decide", content=b"x" * (MAX_BODY_BYTES + 1)).status_code == 413
    assert model.closed


def test_timeout_retains_capacity_and_shutdown_drains() -> None:
    async def scenario() -> None:
        model = ControlledModel()
        app = create_app(lambda: model, timeout=0.02)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                body = {
                    "state": "block",
                    "questions": {
                        "q": Choice(instructions="pick", criteria={"a": "a"}).model_dump()
                    },
                }
                response = await client.post("/v1/decide", json=body)
                assert model.started.is_set()
                assert response.status_code == 504
                assert (await client.post("/v1/decide", json=body)).status_code == 503
                model.release.set()
        assert model.closed

    asyncio.run(scenario())


def body(state: str = "state") -> dict[str, object]:
    return {
        "state": state,
        "questions": {"q": Choice(instructions="pick", criteria={"a": "a"}).model_dump()},
    }


@pytest.mark.parametrize(
    "error,status,code",
    [
        (InputError("too many tokens"), 422, "invalid_request"),
        (ModelClosedError("closed"), 503, "not_ready"),
        (InferenceError("private details"), 500, "inference_failed"),
        (RuntimeError("private details"), 500, "inference_failed"),
    ],
)
def test_failure_mapping_and_capacity_recovery(error: Exception, status: int, code: str) -> None:
    class FailingModel(ControlledModel):
        def decide(self, *, state: str, questions: Mapping[str, Choice]) -> DecisionResult:
            if state == "fail":
                raise error
            return super().decide(state=state, questions=questions)

    model = FailingModel()
    with TestClient(create_app(lambda: model)) as http:
        response = http.post("/v1/decide", json=body("fail"))
        assert response.status_code == status
        assert response.json()["error"]["code"] == code
        assert "private details" not in response.text
        assert http.post("/v1/decide", json=body()).status_code == 200
    assert model.closed


def test_readiness_failure_closes_model() -> None:
    class UnreadyModel(ControlledModel):
        def decide(self, **kwargs: object) -> DecisionResult:
            raise InferenceError("readiness failed")

    model = UnreadyModel()
    with pytest.raises(InferenceError, match="readiness failed"):
        with TestClient(create_app(lambda: model)):
            pytest.fail("startup must fail")
    assert model.closed


def test_cancelled_request_retains_capacity_and_shutdown_waits() -> None:
    async def scenario() -> None:
        model = ControlledModel()
        app = create_app(lambda: model)
        lifetime = app.router.lifespan_context(app)
        await lifetime.__aenter__()
        draining = None
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                pending = asyncio.create_task(client.post("/v1/decide", json=body("block")))
                assert await asyncio.to_thread(model.started.wait, 2)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                response = await client.post("/v1/decide", json=body())
                assert response.status_code == 503
                assert response.json()["error"]["code"] == "overloaded"
                draining = asyncio.create_task(lifetime.__aexit__(None, None, None))
                await asyncio.sleep(0)  # Let shutdown reach the active-task drain.
                assert not draining.done()
                assert not model.closed
                assert (await client.get("/health")).status_code == 503
        finally:
            model.release.set()
            if draining is None:
                await lifetime.__aexit__(None, None, None)
            else:
                await asyncio.wait_for(draining, 2)
        assert model.closed

    asyncio.run(scenario())


def test_admission_is_rechecked_after_reading_a_concurrent_body() -> None:
    async def scenario() -> None:
        model = ControlledModel()
        app = create_app(lambda: model)
        entered = [asyncio.Event(), asyncio.Event()]
        send_body = [asyncio.Event(), asyncio.Event()]

        async def stream(index: int) -> AsyncIterator[bytes]:
            import json

            entered[index].set()
            await send_body[index].wait()
            yield json.dumps(body("block")).encode()

        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                tasks = [
                    asyncio.create_task(client.post("/v1/decide", content=stream(i)))
                    for i in range(2)
                ]
                try:
                    for event in entered:
                        await asyncio.wait_for(event.wait(), 2)
                    send_body[0].set()
                    assert await asyncio.to_thread(model.started.wait, 2)
                    send_body[1].set()
                    response = await asyncio.wait_for(tasks[1], 2)
                    assert response.status_code == 503
                    assert response.json()["error"]["code"] == "overloaded"
                finally:
                    for event in send_body:
                        event.set()
                    model.release.set()
                    await asyncio.gather(*tasks)
                assert tasks[0].result().status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("extra,expected", [(-1, 200), (0, 200), (1, 413)])
def test_streamed_body_byte_limit(extra: int, expected: int) -> None:
    async def scenario() -> None:
        import json

        raw = json.dumps(body()).encode()
        raw += b" " * (MAX_BODY_BYTES + extra - len(raw))

        async def stream() -> AsyncIterator[bytes]:
            yield raw[:-1]
            yield raw[-1:]

        app = create_app(ControlledModel)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                assert (await client.post("/v1/decide", content=stream())).status_code == expected

    asyncio.run(scenario())
