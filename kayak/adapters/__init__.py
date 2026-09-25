"""Borrow Laya or Jev execution through Kayak's named-question Python interface.

The caller constructs and closes the provider object. Importing this module does
not import either SDK, load a model, or open a connection. Provider errors propagate;
Kayak adds no retries, scheduling, configuration, or resource ownership.
"""

from collections.abc import Mapping
from inspect import iscoroutinefunction
from typing import Protocol, cast

from .. import judgments
from ..judgments import JudgmentQuestion, JudgmentRequest
from ._results import (
    ProviderAnswer,
    ProviderChoice,
    ProviderNoul,
    ProviderResult,
    ProviderScore,
    decode,
)

__all__ = [
    "AsyncJev",
    "Jev",
    "Laya",
    "ProviderAnswer",
    "ProviderChoice",
    "ProviderNoul",
    "ProviderResult",
    "ProviderScore",
]


class _Laya(Protocol):
    def predict(self, state: str, questions: dict[str, dict[str, object]]) -> object: ...


class _HTTPBody(Protocol):
    @property
    def content(self) -> bytes: ...


class _JevResponse(Protocol):
    @property
    def raw_http_response(self) -> _HTTPBody: ...


class _JevCall(Protocol):
    def __call__(self, *, state: str, questions: dict[str, dict[str, object]]) -> _JevResponse: ...


class _AsyncJevCall(Protocol):
    async def __call__(
        self, *, state: str, questions: dict[str, dict[str, object]]
    ) -> _JevResponse: ...


def _questions(request: JudgmentRequest) -> dict[str, dict[str, object]]:
    # Send original typed questions, not CLM-compiled false/true candidate text.
    return {
        name: question.model_dump(mode="json", exclude_none=True)
        for name, question in request.questions.items()
    }


class Laya:
    """Wrap an already constructed Laya Agent or Router; its settings remain authoritative."""

    def __init__(self, agent: _Laya) -> None:
        self._agent = agent

    def judge(
        self, *, state: str, questions: Mapping[str, JudgmentQuestion | Mapping[str, object]]
    ) -> ProviderResult:
        request = judgments.request_from(state, questions)
        response = self._agent.predict(state=request.state, questions=_questions(request))
        return decode("laya", request, response)


class Jev:
    """Wrap a caller-owned TypeSafeClient with its configured model, retry and timeout policy."""

    def __init__(self, client: object) -> None:
        # A narrow dynamic SDK boundary avoids importing its optional dependencies
        # or duplicating its full TypedDict input hierarchy. Responses are checked.
        call = getattr(client, "system_one", None)
        if not callable(call) or iscoroutinefunction(call):
            raise TypeError("Jev requires a synchronous TypeSafeClient with system_one")
        self._call = cast(_JevCall, call)

    def judge(
        self, *, state: str, questions: Mapping[str, JudgmentQuestion | Mapping[str, object]]
    ) -> ProviderResult:
        request = judgments.request_from(state, questions)
        response = self._call(state=request.state, questions=_questions(request))
        return decode("jev", request, response.raw_http_response.content)


class AsyncJev:
    """Await a caller-owned AsyncTypeSafeClient directly, including cancellation."""

    def __init__(self, client: object) -> None:
        call = getattr(client, "system_one", None)
        if not callable(call) or not iscoroutinefunction(call):
            raise TypeError("AsyncJev requires an AsyncTypeSafeClient with system_one")
        self._call = cast(_AsyncJevCall, call)

    async def judge(
        self, *, state: str, questions: Mapping[str, JudgmentQuestion | Mapping[str, object]]
    ) -> ProviderResult:
        request = judgments.request_from(state, questions)
        response = await self._call(state=request.state, questions=_questions(request))
        return decode("jev", request, response.raw_http_response.content)
