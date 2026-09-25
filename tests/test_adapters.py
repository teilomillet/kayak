"""Foreign execution preserves question meaning, native evidence, and caller ownership."""

import asyncio
import copy
import json
import subprocess
import sys
import traceback
from collections.abc import Callable
from typing import assert_type

import pytest
from test_judgments import mixed_request

from kayak import InferenceError, InputError, Noul
from kayak.adapters import AsyncJev, Jev, Laya, ProviderResult


def provider_body(*, rounded: bool = False) -> dict[str, object]:
    return {
        "model": "provider-fixture",
        "usage": {"input_tokens": 17, "output_tokens": 0},
        "answers": {
            "route": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"a": 0.5, "b": 0.5},
                "confidence": 0.0,
            },
            "urgent": {"type": "noul", "noul": 0.8123},
            "severity": {
                "type": "score",
                "score": 1.0 if rounded else 1.25,
                "legend": {"0": "Low", "1": "Medium", "2": "High"},
                "probabilities": (
                    {"0": 0.3333, "1": 0.3333, "2": 0.3333}
                    if rounded
                    else {"0": 0.25, "1": 0.25, "2": 0.5}
                ),
                "confidence": 0.12,
            },
        },
        "routing": {"model": "fixture", "reason": "controlled test"},
    }


class Body:
    def __init__(self, body: dict[str, object]) -> None:
        self.content = json.dumps(body).encode()


class Response:
    def __init__(self, body: dict[str, object]) -> None:
        self.raw_http_response = Body(body)


class Provider:
    def __init__(self, body: dict[str, object] | None = None) -> None:
        self.body = body if body is not None else provider_body()
        self.calls: list[tuple[str, dict[str, dict[str, object]]]] = []
        self.change: Callable[[], None] = lambda: None

    def predict(self, state: str, questions: dict[str, dict[str, object]]) -> object:
        self.calls.append((state, copy.deepcopy(questions)))
        self.change()
        questions.clear()  # The provider may mutate its independent input copy.
        return self.body

    def system_one(self, *, state: str, questions: dict[str, dict[str, object]]) -> Response:
        self.predict(state, questions)
        return Response(self.body)

    def close(self) -> None:
        pytest.fail("the adapter must not close a borrowed provider")


@pytest.mark.parametrize("adapter", [Laya, Jev])
def test_mixed_questions_pass_verbatim_and_results_keep_native_evidence(
    adapter: type[Laya] | type[Jev],
) -> None:
    provider = Provider()
    request = mixed_request()
    judge = adapter(provider)
    result = judge.judge(state=request.state, questions=request.questions)
    assert_type(result, ProviderResult)
    assert provider.calls == [
        (
            request.state,
            {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"a": "billing", "b": "support"},
                },
                "urgent": {"type": "noul", "instructions": "  Is this urgent?\n"},
                "severity": {
                    "type": "score",
                    "instructions": "Assess impact",
                    "criteria": ["Low", "Medium", "High"],
                },
            },
        )
    ]
    choice, noul, score = (result.answers[key] for key in request.questions)
    assert choice.type == "choice" and choice.choice == "b"  # Keep provider tie policy.
    assert noul.type == "noul" and noul.noul == 0.8123 and noul.probabilities is None
    assert score.type == "score" and score.score == 1.25
    assert result.calibration == "unknown" and result.input_tokens == 17
    assert result.raw == provider.body
    assert not hasattr(choice, "scores") and not hasattr(choice, "confidence")
    assert ProviderResult.model_validate_json(result.model_dump_json()) == result
    # A retained result owns its evidence, even if the provider reuses output containers.
    provider.body.clear()
    assert result.raw["model"] == "provider-fixture"
    assert choice.probabilities is not None
    original = copy.deepcopy(result.raw)
    choice.probabilities.clear()
    assert result.raw == original


def test_laya_rounding_is_preserved_and_not_silently_normalized() -> None:
    request = mixed_request()
    result = Laya(Provider(provider_body(rounded=True))).judge(
        state=request.state, questions=request.questions
    )
    score = result.answers["severity"]
    assert score.type == "score" and score.score == 1.0
    assert score.probabilities == {"0": 0.3333, "1": 0.3333, "2": 0.3333}
    assert sum(score.probabilities.values()) == 0.9999


def test_absent_model_usage_and_distributions_stay_unknown() -> None:
    result = Laya(Provider({"answers": {"n": {"type": "noul", "noul": 0.5}}})).judge(
        state="text", questions={"n": Noul(instructions="True?")}
    )
    assert result.model is None and result.input_tokens is None
    assert result.answers["n"].probabilities is None


@pytest.mark.parametrize("adapter", [Laya, Jev])
def test_custom_criteria_are_not_compiled_to_clm_text(
    adapter: type[Laya] | type[Jev],
) -> None:
    provider = Provider({"answers": {"n": {"type": "noul", "noul": 0.5}}})
    questions = {"n": Noul(instructions="Vrai?", criteria={"true": " Café ", "false": ""})}
    adapter(provider).judge(state="é", questions=questions)
    assert provider.calls[0][1]["n"]["criteria"] == {"true": " Café ", "false": ""}


@pytest.mark.parametrize("adapter", [Laya, Jev])
def test_request_is_snapshotted_before_provider_effects(adapter: type[Laya] | type[Jev]) -> None:
    provider = Provider()
    request = mixed_request()
    provider.change = request.questions.clear
    result = adapter(provider).judge(state=request.state, questions=request.questions)
    assert list(result.answers) == ["route", "urgent", "severity"]


@pytest.mark.parametrize("adapter", [Laya, Jev])
def test_invalid_request_fails_before_execution(adapter: type[Laya] | type[Jev]) -> None:
    provider = Provider()
    with pytest.raises(InputError):
        adapter(provider).judge(state="", questions={"n": Noul(instructions="True?")})
    request = mixed_request()
    score = request.questions["severity"]
    assert score.type == "score"
    score.criteria.clear()
    with pytest.raises(InputError):
        adapter(provider).judge(state=request.state, questions=request.questions)
    assert provider.calls == []


def set_field(body: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target = body
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value


@pytest.mark.parametrize("adapter", [Laya, Jev])
@pytest.mark.parametrize(
    "path,value",
    [
        (("answers",), {}),
        (("answers", "unknown"), {"type": "noul", "noul": 0.5}),
        (("answers", "route", "choice"), "private-invalid-label"),
        (("answers", "route", "type"), "score"),
        (("answers", "route", "probabilities"), {"a": 0.9, "b": 0.1}),
        (("answers", "route", "probabilities"), {"a": 1.0}),
        (("answers", "route", "probabilities"), {"a": -0.5, "b": 1.5}),
        (("answers", "urgent", "noul"), "0.5"),
        (("answers", "urgent", "noul"), float("nan")),
        (("answers", "urgent", "noul"), True),
        (("answers", "severity", "score"), 3.0),
        (("answers", "severity", "legend"), {"0": "changed", "1": "Medium", "2": "High"}),
        (("usage", "input_tokens"), -1),
        (("usage", "input_tokens"), True),
        (("model",), 12),
    ],
)
def test_invalid_response_is_rejected_without_leaking_body(
    adapter: type[Laya] | type[Jev],
    path: tuple[str, ...],
    value: object,
) -> None:
    body = provider_body()
    set_field(body, path, value)
    body["private"] = "private-input-do-not-log"
    request = mixed_request()
    with pytest.raises(InferenceError) as captured:
        adapter(Provider(body)).judge(state=request.state, questions=request.questions)
    assert "private-" not in "".join(traceback.format_exception(captured.value))


@pytest.mark.parametrize(
    "path,value",
    [
        (("answers", "route", "probabilities"), {"a": 0.0, "b": 0.0}),
        (("answers", "urgent", "probabilities"), {"false": 0.5, "true": 0.5}),
        (("answers", "severity", "score"), 0.0),
    ],
)
def test_laya_arithmetic_uses_its_documented_precision(
    path: tuple[str, ...],
    value: object,
) -> None:
    body = provider_body()
    set_field(body, path, value)
    request = mixed_request()
    with pytest.raises(InferenceError):
        Laya(Provider(body)).judge(state=request.state, questions=request.questions)


def test_jev_does_not_invent_a_precision_contract() -> None:
    body = provider_body(rounded=True)
    request = mixed_request()
    result = Jev(Provider(body)).judge(state=request.state, questions=request.questions)
    score = result.answers["severity"]
    assert score.type == "score" and score.score == 1.0
    assert score.probabilities == {"0": 0.3333, "1": 0.3333, "2": 0.3333}
    assert result.raw == body and result.calibration == "unknown"


@pytest.mark.parametrize("adapter", [Laya, Jev])
def test_provider_errors_propagate_without_added_retries(adapter: type[Laya] | type[Jev]) -> None:
    provider = Provider()
    error = RuntimeError("provider error")

    def fail() -> None:
        raise error

    provider.change = fail
    request = mixed_request()
    with pytest.raises(RuntimeError) as captured:
        adapter(provider).judge(state=request.state, questions=request.questions)
    assert captured.value is error and len(provider.calls) == 1


class AsyncProvider:
    def __init__(self) -> None:
        self.provider = Provider()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def system_one(self, *, state: str, questions: dict[str, dict[str, object]]) -> Response:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.provider.system_one(state=state, questions=questions)


def test_async_snapshots_before_await_and_preserves_cancellation() -> None:
    async def run() -> None:
        provider = AsyncProvider()
        request = mixed_request()
        task = asyncio.create_task(
            AsyncJev(provider).judge(
                state=request.state,
                questions=request.questions,
            )
        )
        await provider.started.wait()
        request.questions.clear()
        provider.release.set()
        result = await task
        assert len(result.answers) == 3 and len(provider.provider.calls) == 1

        provider = AsyncProvider()
        request = mixed_request()
        task = asyncio.create_task(
            AsyncJev(provider).judge(
                state=request.state,
                questions=request.questions,
            )
        )
        await provider.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.cancelled and not provider.provider.calls

    asyncio.run(run())


def test_async_validation_and_provider_failure() -> None:
    async def run() -> None:
        provider = AsyncProvider()
        with pytest.raises(InputError):
            await AsyncJev(provider).judge(state="", questions={})
        assert not provider.started.is_set()
        provider.release.set()
        provider.provider.body = {"answers": {}}
        request = mixed_request()
        with pytest.raises(InferenceError):
            await AsyncJev(provider).judge(state=request.state, questions=request.questions)
        error = TimeoutError("provider timeout")

        def fail() -> None:
            raise error

        provider.provider.change = fail
        with pytest.raises(TimeoutError) as captured:
            await AsyncJev(provider).judge(state=request.state, questions=request.questions)
        assert captured.value is error and len(provider.provider.calls) == 2

    asyncio.run(run())


@pytest.mark.parametrize("adapter", [Jev, AsyncJev])
def test_invalid_client_fails_at_construction(adapter: type[Jev] | type[AsyncJev]) -> None:
    with pytest.raises(TypeError, match="system_one"):
        adapter(object())


def test_wrong_sync_async_client_is_rejected_before_execution() -> None:
    with pytest.raises(TypeError, match="synchronous"):
        Jev(AsyncProvider())
    with pytest.raises(TypeError, match="AsyncTypeSafeClient"):
        AsyncJev(Provider())


def test_import_does_not_load_optional_sdks_or_inference_libraries() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import kayak.adapters; "
                "assert not {'laya', 'typesafe_sdk', 'torch', 'transformers'} & sys.modules.keys()"
            ),
        ],
        check=True,
    )
