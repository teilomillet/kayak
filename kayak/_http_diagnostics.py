"""HTTP request identity and optional completion events, independent of inference lifetime."""

import asyncio
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from starlette.requests import ClientDisconnect, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ._diagnostics import emit


@dataclass
class RequestContext:
    request_id: str
    code: str | None = None


def request_context(request: Request) -> RequestContext:
    context: object = request.scope["kayak.request_context"]
    assert isinstance(context, RequestContext)
    return context


class RequestDiagnostics:
    def __init__(self, app: ASGIApp, *, enabled: bool) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        context = RequestContext(uuid4().hex)
        scope["kayak.request_context"] = context
        started = perf_counter() if self.enabled else 0.0
        status: int | None = None
        completed = False
        event = "http.failed"
        error: Exception | None = None

        async def send_response(message: Message) -> None:
            nonlocal status, completed
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ] + [(b"x-request-id", context.request_id.encode("ascii"))]
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            await self.app(scope, receive, send_response)
        except ClientDisconnect:
            event = "http.disconnected"
            raise
        except asyncio.CancelledError:
            event = "http.cancelled"
            raise
        except Exception as exc:
            error = exc
            raise
        finally:
            if self.enabled:
                # A fixed route name avoids recording arbitrary paths or query strings.
                path: object = getattr(scope.get("route"), "path", None)
                route = (
                    path
                    if isinstance(path, str) and path in {"/health", "/v1/model", "/v1/decide"}
                    else "other"
                )
                emit(
                    "http.completed" if completed else event,
                    request_id=context.request_id,
                    duration=perf_counter() - started,
                    status_code=status,
                    code=context.code,
                    route=route,
                    error=error,
                )
