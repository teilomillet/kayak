"""Async HTTP preserves sync contracts while yielding and releasing resources."""

import asyncio
from collections.abc import Mapping

import httpx
import pytest
from test_contract import sample_result
from test_server import ControlledModel

from kayak import (
    AsyncClient,
    Choice,
    Client,
    DecisionRequest,
    InputError,
    Noul,
    RemoteError,
    TransportError,
)
from kayak.server import create_app


def test_async_call_yields_snapshots_inputs_and_closes(questions: dict[str, Choice]) -> None:
    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        original = DecisionRequest(state="charged twice", questions=questions)

        class Transport(httpx.AsyncBaseTransport):
            closed = False
            calls = 0

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                self.calls += 1
                assert request.headers["authorization"] == "Bearer test-key"
                assert request.url.path == "/prefix/v1/decide"
                assert request.headers["content-type"] == "application/json"
                assert request.extensions["timeout"] == dict.fromkeys(
                    ("connect", "read", "write", "pool"), 120.0
                )
                entered.set()
                await release.wait()
                assert DecisionRequest.model_validate_json(request.content) == original
                return httpx.Response(200, content=sample_result().model_dump_json())

            async def aclose(self) -> None:
                self.closed = True

        transport = Transport()
        async with AsyncClient(
            base_url="http://fixture.test/prefix", api_key="test-key", transport=transport
        ) as client:
            pending = asyncio.create_task(client.decide(state="charged twice", questions=questions))
            try:
                # Another coroutine runs while network I/O is suspended.
                await asyncio.wait_for(entered.wait(), 2)
                assert not pending.done()
                questions["route"].criteria.clear()
            finally:
                release.set()
            assert await asyncio.wait_for(pending, 2) == sample_result()
        assert transport.closed and transport.calls == 1
        await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("dictionary", [False, True])
def test_async_decisions_and_model_info_match_sync(
    dictionary: bool, questions: dict[str, Choice]
) -> None:
    inputs: Mapping[str, Choice | Mapping[str, object]] = questions
    if dictionary:
        inputs = {name: question.model_dump() for name, question in questions.items()}

    def respond(request: httpx.Request) -> httpx.Response:
        result = sample_result()
        value = result.model if request.method == "GET" else result
        return httpx.Response(
            200, content=value.model_dump_json(), headers={"X-Future-Header": "allowed"}
        )

    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as sync:
        expected = sync.decide(state="charged twice", questions=inputs)
        info = sync.model_info()

    async def scenario() -> None:
        async with AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(respond)
        ) as client:
            assert await client.decide(state="charged twice", questions=inputs) == expected
            assert await client.model_info() == info

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 422, 500, 503, 504, 307, 0])
@pytest.mark.parametrize("operation", ["decide", "model_info", "rank", "judge"])
def test_async_errors_are_typed_have_request_ids_and_are_not_retried(
    status: int, operation: str, questions: dict[str, Choice]
) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if status == 0:
            raise httpx.ReadTimeout("private details", request=request)
        code = "invalid_request" if status == 422 else "fixture"
        return httpx.Response(
            status,
            json={"error": {"code": code, "message": "failure"}},
            headers={"x-request-id": "test-123", "location": "https://elsewhere.test"},
        )

    async def scenario() -> None:
        async with AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(respond)
        ) as client:
            with pytest.raises(InputError):
                await client.decide(state=" ", questions=questions)
            with pytest.raises(InputError):
                await client.rank(state="state", instructions="Choose", candidates={})
            assert calls == []
            expected = (
                InputError if status == 422 else (TransportError if status == 0 else RemoteError)
            )
            with pytest.raises(expected) as error:
                if operation == "decide":
                    await client.decide(state="charged twice", questions=questions)
                elif operation == "rank":
                    await client.rank(
                        state="charged twice", instructions="Which team?", candidates={"a": "a"}
                    )
                elif operation == "judge":
                    await client.judge(
                        state="charged twice", questions={"q": Noul(instructions="True?")}
                    )
                else:
                    await client.model_info()
            assert error.value.request_id == (None if status == 0 else "test-123")
            assert "private details" not in str(error.value)
            if isinstance(error.value, RemoteError):
                assert error.value.status_code == status
        assert len(calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["extra", "ids", "order", "distribution", "model"])
def test_async_rejects_malformed_results(change: str, questions: dict[str, Choice]) -> None:
    payload = sample_result().model_dump()
    if change == "extra":
        payload["extra"] = True
    elif change == "ids":
        payload["answers"]["wrong"] = payload["answers"].pop("route")
    elif change == "order":
        answer = payload["answers"]["route"]
        answer["scores"] = dict(reversed(list(answer["scores"].items())))
    elif change == "distribution":
        payload["answers"]["route"]["probabilities"] = {"billing": 0.5, "support": 0.5}
    else:
        payload["model"]["extra"] = True

    async def scenario() -> None:
        async with AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=payload, headers={"x-request-id": "invalid-1"})
            ),
        ) as client:
            with pytest.raises(TransportError) as error:
                await client.decide(state="charged twice", questions=questions)
            assert error.value.request_id == "invalid-1"

    asyncio.run(scenario())


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
@pytest.mark.parametrize(
    ("base_url", "api_key", "timeout"),
    [
        ("ftp://test", None, 1.0),
        ("http://user:secret@test", None, 1.0),
        ("http://test", "private-é", 1.0),
        ("http://test", None, float("nan")),
        ("http://test", None, 0.0),
    ],
)
def test_client_configuration_validation_is_shared(
    client_type: type[Client] | type[AsyncClient],
    base_url: str,
    api_key: str | None,
    timeout: float,
) -> None:
    with pytest.raises(InputError):
        client_type(base_url=base_url, api_key=api_key, timeout=timeout)


def test_cancellation_retains_server_capacity_and_shutdown_drains() -> None:
    async def scenario() -> None:
        model = ControlledModel()
        app = create_app(lambda: model)
        questions = {"q": Choice(instructions="Pick", criteria={"a": "a"})}
        lifetime = app.router.lifespan_context(app)
        await lifetime.__aenter__()
        draining = None
        try:
            async with AsyncClient(
                base_url="http://test", transport=httpx.ASGITransport(app)
            ) as client:
                pending = asyncio.create_task(client.decide(state="block", questions=questions))
                assert await asyncio.to_thread(model.started.wait, 2)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                with pytest.raises(RemoteError) as error:
                    await client.decide(state="state", questions=questions)
                assert error.value.status_code == 503 and error.value.code == "overloaded"
                draining = asyncio.create_task(lifetime.__aexit__(None, None, None))
                await asyncio.sleep(0)
                assert not draining.done() and not model.closed
        finally:
            model.release.set()
            if draining is None:
                await lifetime.__aexit__(None, None, None)
            else:
                await asyncio.wait_for(draining, 2)
        assert model.closed

    asyncio.run(scenario())
