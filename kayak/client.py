"""Sync and async HTTP clients with one request, response, and error policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import TracebackType
from typing import Literal, Self

import httpx
from pydantic import TypeAdapter, ValidationError

from . import decisions, judgments, ranking
from .decisions import Choice, DecisionRequest, DecisionResult, ModelInfo
from .errors import InferenceError, InputError, RemoteError, TransportError
from .judgments import JudgmentQuestion, JudgmentResult
from .ranking import RankingResult

# Reuse the typed serializer; HTTP consumes bytes, so avoid JSON text round trips.
_request_json = TypeAdapter(decisions.DecisionRequest)


class Client:
    """A synchronous connection to a Kayak service, with validated typed results.

    Use a context manager to close connections. Construction loads no model;
    model_info() and decide() perform explicit HTTP requests without retries.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        _validate_settings(base_url, api_key, timeout)
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            follow_redirects=False,
            transport=transport,
        )

    def decide(
        self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        """Evaluate named Choice questions; return checked answers without automatic retries."""
        request = decisions.request_from(state, questions)
        response = self._send("POST", "v1/decide", content=_request_json.dump_json(request))
        return _decision_result(request, response)

    def model_info(self) -> ModelInfo:
        """Fetch the server's model identity and device without running a decision."""
        response = self._send("GET", "v1/model")
        return _model_info(response)

    def judge(
        self, *, state: str, questions: Mapping[str, JudgmentQuestion | Mapping[str, object]]
    ) -> JudgmentResult:
        """Evaluate Choice, Noul, and Score through one compatible /v1 Choice request."""
        request = judgments.request_from(state, questions)
        decision = request.as_decision()
        result = self.decide(state=decision.state, questions=decision.questions)
        return request.decode(result)

    def rank(
        self, *, state: str, instructions: str, candidates: Mapping[str, str]
    ) -> RankingResult:
        """Rank candidates via /v1/decide, including on an older Choice-only server."""
        request = ranking.request_from(state, instructions, candidates)
        result = self.decide(state=request.state, questions=request.questions)
        return ranking.result_from(result)

    def _send(
        self, method: Literal["GET", "POST"], path: str, *, content: bytes | None = None
    ) -> httpx.Response:
        """Keep network failures and server error translation identical across operations."""
        try:
            response = self._http.request(
                method,
                path,
                content=content,
                headers={"Content-Type": "application/json"} if content is not None else {},
            )
        except httpx.HTTPError as exc:
            raise TransportError(
                f"Kayak request failed ({type(exc).__name__}); not retried"
            ) from exc
        _check_status(response)
        return response

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class AsyncClient:
    """An asynchronous HTTP connection to a Kayak service, without local inference.

    Use ``async with`` or await ``aclose()`` to release connections. Cancellation
    stops waiting for HTTP; it does not stop the server's active computation.
    Calls make one attempt, and occupied service capacity still returns HTTP 503.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        _validate_settings(base_url, api_key, timeout)
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            follow_redirects=False,
            transport=transport,
        )

    async def decide(
        self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        """Validate and snapshot input before I/O; await one checked Choice response."""
        request = decisions.request_from(state, questions)
        response = await self._send("POST", "v1/decide", content=_request_json.dump_json(request))
        return _decision_result(request, response)

    async def model_info(self) -> ModelInfo:
        """Fetch model identity and device without requesting a decision."""
        response = await self._send("GET", "v1/model")
        return _model_info(response)

    async def judge(
        self, *, state: str, questions: Mapping[str, JudgmentQuestion | Mapping[str, object]]
    ) -> JudgmentResult:
        """Await one Choice request, then decode Noul/Score without changing server semantics."""
        request = judgments.request_from(state, questions)
        decision = request.as_decision()
        result = await self.decide(state=decision.state, questions=decision.questions)
        return request.decode(result)

    async def rank(
        self, *, state: str, instructions: str, candidates: Mapping[str, str]
    ) -> RankingResult:
        """Rank candidates through the same /v1 Choice request as the sync client."""
        request = ranking.request_from(state, instructions, candidates)
        result = await self.decide(state=request.state, questions=request.questions)
        return ranking.result_from(result)

    async def _send(
        self, method: Literal["GET", "POST"], path: str, *, content: bytes | None = None
    ) -> httpx.Response:
        try:
            response = await self._http.request(
                method,
                path,
                content=content,
                headers={"Content-Type": "application/json"} if content is not None else {},
            )
        except httpx.HTTPError as exc:
            raise TransportError(
                f"Kayak request failed ({type(exc).__name__}); not retried"
            ) from exc
        _check_status(response)
        return response

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _validate_settings(base_url: str, api_key: str | None, timeout: float) -> None:
    """Reject invalid configuration before either client allocates connections."""
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL:
        raise InputError("base_url must be a valid HTTP(S) URL") from None
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
        raise InputError("base_url must be an HTTP(S) URL without embedded credentials")
    if not math.isfinite(timeout) or timeout <= 0:
        raise InputError("timeout must be finite and positive")
    if api_key is not None and not api_key.isascii():
        raise InputError("api_key must contain only ASCII characters for the HTTP header")


def _decision_result(request: DecisionRequest, response: httpx.Response) -> DecisionResult:
    """Validate the result and its relationship to the request at the HTTP boundary."""
    try:
        result = DecisionResult.model_validate_json(response.content)
        decisions.check_result(request, result)
        return result
    except (ValidationError, InferenceError) as exc:
        raise TransportError(
            "Kayak server returned an invalid decision result", request_id=_request_id(response)
        ) from exc


def _model_info(response: httpx.Response) -> ModelInfo:
    try:
        return ModelInfo.model_validate_json(response.content)
    except ValidationError as exc:
        raise TransportError(
            "Kayak server returned invalid model information", request_id=_request_id(response)
        ) from exc


def _check_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    code, message = _error_details(response)
    if response.status_code == 422 and code == "invalid_request":
        raise InputError(message, request_id=_request_id(response))
    raise RemoteError(
        message,
        status_code=response.status_code,
        code=code,
        request_id=_request_id(response),
    )


def _request_id(response: httpx.Response) -> str | None:
    """Accept a bounded printable token; remote headers are untrusted input."""
    value: object = response.headers.get("x-request-id", "")
    if (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value.isascii()
        and all(character.isalnum() or character in "-_." for character in value)
    ):
        return value
    return None


def _error_details(response: httpx.Response) -> tuple[str, str]:
    """Read the server's error code/message, falling back for malformed bodies."""
    fallback = ("http_error", f"Kayak server returned HTTP {response.status_code}")
    try:
        payload: object = response.json()
    except ValueError:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    error: object = payload.get("error")
    if not isinstance(error, dict):
        return fallback
    code: object = error.get("code")
    message: object = error.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        return fallback
    return code, message
