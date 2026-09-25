"""Execute caller-owned RAG callbacks with bounded concurrency and detached evidence.

Callers own backend construction, timeouts, retries, and cleanup. The evaluator
owns only its input snapshots, recorded attempts, and asynchronous worker tasks.
"""

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from time import perf_counter
from typing import Literal

from pydantic import JsonValue

from ._rag import RAGTrace
from ._rag_experiment import (
    RAGAttempt,
    RAGCase,
    RAGDataset,
    RAGEvalConfig,
    RAGFailure,
    RAGInput,
    RAGOutput,
    RAGReport,
    RAGReview,
    RAGReviewInput,
)
from ._rag_reports import score_rag


def _prepare(dataset: RAGDataset, config: RAGEvalConfig | None) -> tuple[RAGDataset, RAGEvalConfig]:
    """Reject invalid experiment data before any user code executes."""
    data = RAGDataset.model_validate(dataset.model_dump())
    settings = RAGEvalConfig.model_validate((config or RAGEvalConfig()).model_dump())
    if any(case.judgments.answer is not None for case in data.cases):
        raise ValueError("live runs require a review callback instead of a prebound answer review")
    return data, settings


def _checked_output(result: RAGTrace | RAGOutput, request: RAGInput) -> RAGOutput:
    """Snapshot observations before handing another callback any mutable values."""
    if isinstance(result, RAGOutput):
        output = RAGOutput.model_validate(result.model_dump())
    elif isinstance(result, RAGTrace):
        output = RAGOutput.model_validate({"final": result.model_dump()})
    else:
        raise TypeError("pipeline must return a RAGTrace or RAGOutput")
    if output.final.id != request.id or output.final.query != request.query:
        raise ValueError("final trace ID and query must match the original input")
    return output


def _checked_review(result: RAGReview, output: RAGOutput, review_input_sha256: str) -> RAGReview:
    if not isinstance(result, RAGReview):
        raise TypeError("review must return a RAGReview")
    reviewed = RAGReview.model_validate(result.model_dump())
    if reviewed.review_input_sha256 != review_input_sha256:
        raise ValueError("review belongs to different inputs, references, or output")
    if (reviewed.correct is not None or reviewed.grounded is not None) and (
        output.final.answer is None
    ):
        raise ValueError("Boolean answer judgments require an observed answer")
    if reviewed.grounded is not None and output.final.context is None:
        raise ValueError("grounding judgments require observed context")
    return reviewed


def _failure(error: Exception, phase: Literal["pipeline", "output", "review"]) -> RAGFailure:
    # The exception message may include document text, credentials, or backend
    # internals. The bounded class name identifies the failure without retaining it.
    return RAGFailure(phase=phase, type=type(error).__name__[:128] or "Exception")


def _run_one(
    case: RAGCase,
    repeat: int,
    system: dict[str, JsonValue],
    pipeline: Callable[[RAGInput], RAGTrace | RAGOutput],
    review: Callable[[RAGReviewInput], RAGReview] | None,
) -> RAGAttempt:
    request = case.input.model_copy(deep=True)
    started = perf_counter()
    try:
        result = pipeline(request)
    except Exception as error:
        duration = perf_counter() - started
        return RAGAttempt(
            input=case.input.model_copy(deep=True),
            repeat=repeat,
            system=deepcopy(system),
            error=_failure(error, "pipeline"),
            duration_seconds=duration,
        )
    duration = perf_counter() - started
    try:
        output = _checked_output(result, case.input)
    except Exception as error:
        return RAGAttempt(
            input=case.input.model_copy(deep=True),
            repeat=repeat,
            system=deepcopy(system),
            error=_failure(error, "output"),
            duration_seconds=duration,
        )
    attempt = RAGAttempt(
        input=case.input.model_copy(deep=True),
        repeat=repeat,
        system=deepcopy(system),
        output=output,
        duration_seconds=duration,
    )
    if review is None:
        return attempt
    request_review = RAGReviewInput(
        case=case.model_copy(deep=True), output=output.model_copy(deep=True)
    )
    review_input_sha256 = request_review.sha256
    started = perf_counter()
    try:
        reviewed = review(request_review)
    except Exception as error:
        review_duration = perf_counter() - started
        return attempt.model_copy(
            update={
                "review_error": _failure(error, "review"),
                "review_duration_seconds": review_duration,
            }
        )
    review_duration = perf_counter() - started
    try:
        checked_review = _checked_review(reviewed, output, review_input_sha256)
    except Exception as error:
        return attempt.model_copy(
            update={
                "review_error": _failure(error, "review"),
                "review_duration_seconds": review_duration,
            }
        )
    return attempt.model_copy(
        update={"review": checked_review, "review_duration_seconds": review_duration}
    )


def evaluate_rag(
    dataset: RAGDataset,
    pipeline: Callable[[RAGInput], RAGTrace | RAGOutput],
    *,
    config: RAGEvalConfig | None = None,
    review: Callable[[RAGReviewInput], RAGReview] | None = None,
    on_attempt: Callable[[RAGAttempt], None] | None = None,
) -> RAGReport:
    """Run cases sequentially; expose failures without sending judgments to a pipeline.

    Each configured repetition is a new invocation, with no evaluator retry or
    cache. Ordinary callback failures become records and do not stop later cases.
    Interrupts propagate. Callback durations exclude validation and scoring. An
    optional hook receives each completed attempt as a detached snapshot, so the
    caller can retain progress before an interruption. Hook failures propagate.
    """
    data, settings = _prepare(dataset, config)
    if settings.max_concurrency != 1:
        raise ValueError("synchronous evaluation requires max_concurrency=1; use aevaluate_rag")
    attempts = []
    for case in data.cases:
        for repeat in range(settings.repeats):
            attempt = _run_one(case, repeat, settings.system, pipeline, review)
            attempts.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt.model_copy(deep=True))
    return score_rag(data, attempts, config=settings)


async def _arun_one(
    case: RAGCase,
    repeat: int,
    system: dict[str, JsonValue],
    pipeline: Callable[[RAGInput], Awaitable[RAGTrace | RAGOutput]],
    review: Callable[[RAGReviewInput], Awaitable[RAGReview]] | None,
) -> RAGAttempt:
    request = case.input.model_copy(deep=True)
    started = perf_counter()
    try:
        result = await pipeline(request)
    except Exception as error:
        duration = perf_counter() - started
        return RAGAttempt(
            input=case.input.model_copy(deep=True),
            repeat=repeat,
            system=deepcopy(system),
            error=_failure(error, "pipeline"),
            duration_seconds=duration,
        )
    duration = perf_counter() - started
    try:
        output = _checked_output(result, case.input)
    except Exception as error:
        return RAGAttempt(
            input=case.input.model_copy(deep=True),
            repeat=repeat,
            system=deepcopy(system),
            error=_failure(error, "output"),
            duration_seconds=duration,
        )
    attempt = RAGAttempt(
        input=case.input.model_copy(deep=True),
        repeat=repeat,
        system=deepcopy(system),
        output=output,
        duration_seconds=duration,
    )
    if review is None:
        return attempt
    request_review = RAGReviewInput(
        case=case.model_copy(deep=True), output=output.model_copy(deep=True)
    )
    review_input_sha256 = request_review.sha256
    started = perf_counter()
    try:
        reviewed = await review(request_review)
    except Exception as error:
        review_duration = perf_counter() - started
        return attempt.model_copy(
            update={
                "review_error": _failure(error, "review"),
                "review_duration_seconds": review_duration,
            }
        )
    review_duration = perf_counter() - started
    try:
        checked_review = _checked_review(reviewed, output, review_input_sha256)
    except Exception as error:
        return attempt.model_copy(
            update={
                "review_error": _failure(error, "review"),
                "review_duration_seconds": review_duration,
            }
        )
    return attempt.model_copy(
        update={"review": checked_review, "review_duration_seconds": review_duration}
    )


async def aevaluate_rag(
    dataset: RAGDataset,
    pipeline: Callable[[RAGInput], Awaitable[RAGTrace | RAGOutput]],
    *,
    config: RAGEvalConfig | None = None,
    review: Callable[[RAGReviewInput], Awaitable[RAGReview]] | None = None,
    on_attempt: Callable[[RAGAttempt], Awaitable[None]] | None = None,
) -> RAGReport:
    """Run asynchronous callbacks with bounded workers and deterministic report order.

    Cancellation stops and awaits every worker owned here. Backend cleanup and
    cancellation of work hidden inside caller callbacks remain the caller's duty.
    Review calls share the worker bound with pipeline calls. No task is allocated
    for each queued input, and configured repeats are retained individually.
    Optional hooks receive detached completed attempts before a worker advances.
    Hooks may run concurrently, share the worker bound, and propagate failures;
    the caller owns persistence and any needed synchronization.
    """
    data, settings = _prepare(dataset, config)
    count = len(data.cases) * settings.repeats
    jobs = iter(
        enumerate((case, repeat) for case in data.cases for repeat in range(settings.repeats))
    )
    attempts: list[RAGAttempt | None] = [None] * count

    async def worker() -> None:
        # Taking the next job has no await, so workers share the iterator safely
        # on this event loop without a lock or a task per input.
        for position, (case, repeat) in jobs:
            attempt = await _arun_one(case, repeat, settings.system, pipeline, review)
            attempts[position] = attempt
            if on_attempt is not None:
                await on_attempt(attempt.model_copy(deep=True))

    workers = [asyncio.create_task(worker()) for _ in range(min(settings.max_concurrency, count))]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    completed = [attempt for attempt in attempts if attempt is not None]
    return score_rag(data, completed, config=settings)
