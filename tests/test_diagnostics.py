"""Observe request identity, privacy, and execution histories at the HTTP boundary."""

import asyncio
import json
import logging
from collections.abc import Mapping

import httpx
import pytest
from fastapi.testclient import TestClient
from test_server import ControlledModel, body

from kayak import Choice, InferenceError
from kayak.decisions import DecisionResult
from kayak.server import create_app


def events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "kayak.diagnostics"
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_ids_and_response_contract_with_logging_on_or_off(
    enabled: bool, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger="kayak.diagnostics")
    if not enabled:

        def unexpected_clock() -> float:
            pytest.fail("disabled diagnostics must not measure request duration")

        monkeypatch.setattr("kayak.server.perf_counter", unexpected_clock)
        monkeypatch.setattr("kayak._http_diagnostics.perf_counter", unexpected_clock)
    model = ControlledModel()
    with TestClient(create_app(lambda: model, diagnostics=enabled, api_key="secret-key")) as http:
        responses = [
            http.get("/health"),
            http.get("/v1/model"),
            http.post("/v1/decide", json=body(), headers={"authorization": "Bearer secret-key"}),
            http.get("/private-path?secret=query"),
        ]
        assert [response.status_code for response in responses] == [200, 401, 200, 404]
        assert responses[0].json() == {"ready": True}
        assert responses[1].json() == {
            "error": {"code": "unauthorized", "message": "a valid bearer token is required"}
        }
        ids = [response.headers["x-request-id"] for response in responses]
        assert len(set(ids)) == 4
        assert all(len(value) == 32 and int(value, 16) >= 0 for value in ids)
        result = DecisionResult.model_validate_json(responses[2].content)
        assert "request_id" not in result.model_dump()
    records = events(caplog)
    if enabled:
        completed = [event for event in records if event["event"] == "http.completed"]
        assert [event["request_id"] for event in completed] == ids
        assert completed[1]["code"] == "unauthorized"
        assert completed[-1]["route"] == "other"
        assert records[0]["event"] == "startup.started"
        assert records[-1]["event"] == "shutdown.completed"
        assert "private-path" not in json.dumps(records)
        assert "secret-key" not in json.dumps(records)
        assert "secret=query" not in json.dumps(records)
    else:
        assert records == []


def test_failure_links_http_to_exception_chain_without_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenModel(ControlledModel):
        def decide(self, *, state: str, questions: Mapping[str, Choice]) -> DecisionResult:
            if state == "private-request-value":
                try:
                    raise ValueError("private-cause-message")
                except ValueError as exc:
                    raise InferenceError("private-wrapper-message") from exc
            return super().decide(state=state, questions=questions)

    caplog.set_level(logging.INFO, logger="kayak.diagnostics")
    with TestClient(create_app(BrokenModel, diagnostics=True)) as http:
        response = http.post(
            "/v1/decide",
            json=body("private-request-value"),
            headers={"x-request-id": "untrusted-caller-id", "traceparent": "private-header"},
        )
        assert response.status_code == 500
        assert http.post("/v1/decide", json=body()).status_code == 200
    records = events(caplog)
    request_id = response.headers["x-request-id"]
    failure = next(
        event
        for event in records
        if event["event"] == "inference.completed" and event.get("request_id") == request_id
    )
    assert failure["code"] == "inference_failed"
    exceptions = failure["exceptions"]
    assert isinstance(exceptions, list)
    assert [entry["type"] for entry in exceptions] == ["InferenceError", "ValueError"]
    assert all(entry["frames"] for entry in exceptions)
    completed = next(
        event
        for event in records
        if event["event"] == "http.completed" and event.get("request_id") == request_id
    )
    assert completed["status_code"] == 500 and completed["code"] == "inference_failed"
    serialized = json.dumps(records)
    for secret in (
        "private-request-value",
        "private-cause-message",
        "private-wrapper-message",
        "untrusted-caller-id",
        "private-header",
    ):
        assert secret not in serialized and secret not in response.text


def test_unhandled_framework_error_has_safe_context_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="kayak.diagnostics")
    app = create_app(ControlledModel, diagnostics=True)

    @app.get("/broken-extension")
    def broken_extension() -> None:
        raise RuntimeError("private-framework-message")

    with TestClient(app, raise_server_exceptions=False) as http:
        response = http.get("/broken-extension")
    assert response.status_code == 500
    # Starlette's outer error middleware produces this response, outside ours.
    assert "x-request-id" not in response.headers
    failure = next(event for event in events(caplog) if event["event"] == "http.failed")
    assert failure["route"] == "other"
    assert "status_code" not in failure
    assert "RuntimeError" in json.dumps(failure["exceptions"])
    assert "private-framework-message" not in json.dumps(events(caplog))


@pytest.mark.parametrize("cancel", [False, True])
def test_http_ends_before_inference_and_capacity_recovers(
    cancel: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="kayak.diagnostics")

    async def scenario() -> None:
        model = ControlledModel()
        app = create_app(lambda: model, timeout=5 if cancel else 0.02, diagnostics=True)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(client.post("/v1/decide", json=body("block")))
                try:
                    assert await asyncio.to_thread(model.started.wait, 2)
                    if cancel:
                        first.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await first
                    else:
                        assert (await first).status_code == 504
                    before = events(caplog)
                    started = next(e for e in before if e["event"] == "inference.started")
                    identity = started["request_id"]
                    assert not any(e["event"] == "inference.completed" for e in before)
                    response = await client.post("/v1/decide", json=body())
                    assert response.status_code == 503
                    assert response.headers["x-request-id"] != identity
                    model.release.set()
                    # A bounded wait observes the computation's completion, not a guessed sleep.
                    async with asyncio.timeout(2):
                        while not any(e["event"] == "inference.completed" for e in events(caplog)):
                            await asyncio.sleep(0.001)
                    assert (await client.post("/v1/decide", json=body())).status_code == 200
                finally:
                    model.release.set()
                    await asyncio.gather(first, return_exceptions=True)
        records = [e for e in events(caplog) if e.get("request_id") == identity]
        assert [e["event"] for e in records] == [
            "inference.started",
            "http.cancelled" if cancel else "http.completed",
            "inference.completed",
        ]
        if cancel:
            assert "status_code" not in records[1]
        else:
            assert records[1]["code"] == "timeout" and records[1]["status_code"] == 504
        assert records[2]["status_code"] == 200
        assert "code" not in records[2]
        assert model.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["load", "warmup"])
def test_startup_failure_is_reported_and_resources_close(
    phase: str, caplog: pytest.LogCaptureFixture
) -> None:
    class UnreadyModel(ControlledModel):
        def decide(self, *, state: str, questions: Mapping[str, Choice]) -> DecisionResult:
            raise InferenceError("private-startup-error")

    model = UnreadyModel()

    def loader() -> ControlledModel:
        if phase == "load":
            raise RuntimeError("private-startup-error")
        return model

    caplog.set_level(logging.INFO, logger="kayak.diagnostics")
    with pytest.raises((RuntimeError, InferenceError)):
        with TestClient(create_app(loader, diagnostics=True)):
            pytest.fail("startup should fail")
    records = events(caplog)
    assert [event["event"] for event in records] == ["startup.started", "startup.failed"]
    assert "private-startup-error" not in json.dumps(records)
    assert model.closed is (phase == "warmup")
