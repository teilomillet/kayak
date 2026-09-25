"""Exercise bounded retries and uncertain outcomes without model weights."""

import math
from collections.abc import Callable

import httpx
import pytest

import kayak
from examples import retry_busy

MODEL = kayak.ModelInfo(
    id="example/simulated",
    revision="fixture-v1",
    fingerprint="simulated-no-weights",
    encoder="none",
    encoder_revision="none",
    device="none",
    dtype="none",
)


def respond(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST" and request.url.path == "/v1/decide"
    payload = kayak.DecisionRequest.model_validate_json(request.content)
    answers: dict[str, kayak.ChoiceAnswer] = {}
    for name, question in payload.questions.items():
        choice = next(iter(question.criteria))
        scores = {key: float(key == choice) for key in question.criteria}
        denominator = sum(math.exp(score) for score in scores.values())
        answers[name] = kayak.ChoiceAnswer(
            choice=choice,
            scores=scores,
            probabilities={key: math.exp(score) / denominator for key, score in scores.items()},
        )
    result = kayak.DecisionResult(model=MODEL, answers=answers, input_tokens=0)
    return httpx.Response(200, content=result.model_dump_json())


def use_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    original = kayak.Client

    def client(*, base_url: str, api_key: str | None) -> kayak.Client:
        return original(base_url=base_url, api_key=api_key, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(kayak, "Client", client)
    monkeypatch.setenv("KAYAK_BASE_URL", "http://example.test")
    monkeypatch.delenv("KAYAK_API_KEY", raising=False)


@pytest.mark.parametrize("busy_responses", [0, 1, 2, 3])
def test_busy_retries_are_bounded_and_preserve_the_request(
    busy_responses: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[bytes] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.content)
        if len(requests) <= busy_responses:
            return httpx.Response(503, json={"error": {"code": "overloaded", "message": "busy"}})
        return respond(request)

    use_transport(monkeypatch, handler)
    monkeypatch.setattr(retry_busy, "sleep", sleeps.append)
    if busy_responses == 3:
        with pytest.raises(kayak.RemoteError):
            retry_busy.main()
    else:
        retry_busy.main()
        result = kayak.DecisionResult.model_validate_json(capsys.readouterr().out)
        assert result.model == MODEL
    assert len(requests) == min(busy_responses + 1, 3)
    assert len(set(requests)) == 1
    assert sleeps == [0.25, 0.5][:busy_responses]


@pytest.mark.parametrize(
    "status,code", [(504, "timeout"), (503, "not_ready"), (401, "unauthorized"), (0, "network")]
)
def test_unknown_completion_and_non_busy_errors_are_not_retried(
    status: int,
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if status == 0:
            raise httpx.ReadTimeout("unknown completion", request=request)
        return httpx.Response(status, json={"error": {"code": code, "message": "failure"}})

    use_transport(monkeypatch, handler)
    with pytest.raises(kayak.KayakError):
        retry_busy.main()
    assert calls == 1
