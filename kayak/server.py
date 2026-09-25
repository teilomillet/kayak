"""Single-model HTTP transport with bounded admission and explicit ownership."""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from . import decisions
from ._diagnostics import emit
from ._http_diagnostics import RequestDiagnostics, request_context
from .decisions import Choice, DecisionRequest, DecisionResult, ModelInfo
from .errors import InferenceError, InputError, ModelClosedError

MAX_BODY_BYTES = decisions.MAX_REQUEST_BYTES
logger = logging.getLogger("kayak.server")


class DecisionModel(Protocol):
    """The model operations owned by the HTTP service, including its test doubles."""

    @property
    def info(self) -> ModelInfo: ...

    def decide(self, *, state: str, questions: Mapping[str, Choice]) -> DecisionResult: ...

    def close(self) -> None: ...


def failure(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(
    loader: Callable[[], DecisionModel],
    *,
    api_key: str | None = None,
    timeout: float = 120.0,
    diagnostics: bool = False,
) -> FastAPI:
    """Own one loaded model for the service lifetime, with zero queued requests.

    Timed-out or disconnected requests retain capacity until inference finishes.
    Shutdown drains active inference before closing the model.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if api_key is not None and not api_key:
        raise ValueError("api_key must not be empty")
    model: DecisionModel | None = None
    active: asyncio.Task[tuple[Response, str | None]] | None = None
    accepting = False

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal model, accepting
        started = perf_counter() if diagnostics else 0.0
        if diagnostics:
            emit("startup.started")
        try:
            model = await asyncio.to_thread(loader)
            # Readiness exercises the loaded model; merely opening a port is insufficient.
            await asyncio.to_thread(
                model.decide,
                state="Ready.",
                questions={
                    "ready": Choice(instructions="Select readiness.", criteria={"ready": "Ready."})
                },
            )
            accepting = True
            if diagnostics:
                emit(
                    "startup.ready",
                    duration=perf_counter() - started,
                    fingerprint=model.info.fingerprint,
                )
        except BaseException as exc:
            if diagnostics:
                emit("startup.failed", duration=perf_counter() - started, error=exc)
            if model is not None:
                await asyncio.to_thread(model.close)
                model = None
            raise
        try:
            yield
        finally:
            accepting = False
            if diagnostics:
                emit("shutdown.started")
            if active is not None:
                await asyncio.gather(active, return_exceptions=True)
            try:
                await asyncio.to_thread(model.close)
            except Exception as exc:
                if diagnostics:
                    emit("shutdown.failed", error=exc)
                raise
            model = None
            if diagnostics:
                emit("shutdown.completed")

    app = FastAPI(title="Kayak", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(RequestDiagnostics, enabled=diagnostics)

    def reject(request: Request, status: int, code: str, message: str) -> JSONResponse:
        request_context(request).code = code
        return failure(status, code, message)

    def authorized(request: Request) -> bool:
        return api_key is None or hmac.compare_digest(
            request.headers.get("authorization", "").encode(), f"Bearer {api_key}".encode()
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ready": accepting}, status_code=200 if accepting else 503)

    @app.get("/v1/model")
    async def model_info(request: Request) -> Response:
        if not authorized(request):
            return reject(request, 401, "unauthorized", "a valid bearer token is required")
        if not accepting or model is None:
            return reject(request, 503, "not_ready", "model is not ready")
        return Response(model.info.model_dump_json(), media_type="application/json")

    async def evaluate(body: DecisionRequest, request_id: str) -> tuple[Response, str | None]:
        started = perf_counter() if diagnostics else 0.0
        code: str | None = None
        error: Exception | None = None
        if diagnostics:
            emit("inference.started", request_id=request_id)
        try:
            assert model is not None  # Admission and shutdown own this lifetime.
            result = await asyncio.to_thread(
                model.decide, state=body.state, questions=body.questions
            )
            response: Response = Response(result.model_dump_json(), media_type="application/json")
        except InputError as exc:
            code, error = "invalid_request", exc
            response = failure(422, code, str(exc))
        except ModelClosedError as exc:
            code, error = "not_ready", exc
            response = failure(503, code, "model is not ready")
        except Exception as exc:
            code, error = "inference_failed", exc
            if not diagnostics:
                logger.error(
                    "model inference failed"
                    if isinstance(exc, InferenceError)
                    else "unexpected model failure"
                )
            response = failure(500, code, "model inference failed")
        if diagnostics:
            emit(
                "inference.completed",
                request_id=request_id,
                duration=perf_counter() - started,
                status_code=response.status_code,
                code=code,
                error=error,
                fingerprint=model.info.fingerprint if model is not None else None,
            )
        return response, code

    @app.post("/v1/decide")
    async def decide(request: Request) -> Response:
        nonlocal active
        if not authorized(request):
            return reject(request, 401, "unauthorized", "a valid bearer token is required")
        if not accepting:
            return reject(request, 503, "not_ready", "model is not ready")
        if active is not None and not active.done():
            return reject(request, 503, "overloaded", "model is busy; no request was queued")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > MAX_BODY_BYTES:
                return reject(request, 413, "request_too_large", "request body exceeds 1 MiB")
            raw.extend(chunk)
        try:
            body = DecisionRequest.model_validate_json(raw)
        except ValidationError as exc:
            return reject(request, 422, "invalid_request", decisions.validation_message(exc))
        # Body reading awaits I/O, so recheck admission before starting work.
        if active is not None and not active.done():
            return reject(request, 503, "overloaded", "model is busy; no request was queued")
        context = request_context(request)
        active = asyncio.create_task(evaluate(body, context.request_id))
        try:
            response, context.code = await asyncio.wait_for(asyncio.shield(active), timeout)
            return response
        except TimeoutError:
            return reject(
                request, 504, "timeout", "inference timed out; computation may still be running"
            )

    return app
