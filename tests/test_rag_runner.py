"""Exercise public callback ownership, error isolation, and concurrent histories."""

import asyncio
from typing import cast

import pytest

from kayak.eval._rag import RAGAnswerReview, RAGContext, RAGJudgments, RAGTrace
from kayak.eval._rag_experiment import (
    RAGAttempt,
    RAGCase,
    RAGDataset,
    RAGEvalConfig,
    RAGInput,
    RAGOutput,
    RAGReview,
    RAGReviewInput,
    RAGScore,
)
from kayak.eval._rag_runner import aevaluate_rag, evaluate_rag
from kayak.eval._ranking import RankedOutput


def _dataset(*identifiers: str) -> RAGDataset:
    return RAGDataset(
        name="recorded tasks",
        cases=[
            RAGCase(
                input=RAGInput(id=identifier, query=f"Question {identifier}", parameters={"n": 1}),
                judgments=RAGJudgments(relevance={"source": 1}),
                reference_answer="independent reference",
            )
            for identifier in identifiers or ("first",)
        ],
    )


def _trace(request: RAGInput) -> RAGTrace:
    return RAGTrace(
        id=request.id,
        query=request.query,
        retrieval=RankedOutput(ids=["source"]),
        context=RAGContext(ids=["source"], text="Observed source text"),
        answer="Observed answer",
        provenance={"model": "first"},
    )


def test_callbacks_receive_separate_owned_inputs_and_return_snapshots() -> None:
    dataset = _dataset()
    outputs: list[RAGTrace] = []
    reviews: list[RAGReview] = []

    def pipeline(request: RAGInput) -> RAGTrace:
        assert set(type(request).model_fields) == {"id", "query", "parameters"}
        assert request.parameters == {"n": 1}
        request.parameters["n"] = 99
        result = _trace(request)
        outputs.append(result)
        return result

    def review(request: RAGReviewInput) -> RAGReview:
        assert request.case.reference_answer == "independent reference"
        assert request.case.input.parameters == {"n": 1}
        digest = request.sha256
        request.case.judgments.relevance.clear()
        request.output.final.provenance.clear()
        reviewed = RAGReview(review_input_sha256=digest, correct=True, grounded=True)
        reviews.append(reviewed)
        return reviewed

    report = evaluate_rag(dataset, pipeline, config=RAGEvalConfig(repeats=2), review=review)
    for output in outputs:
        output.provenance.clear()
    for reviewed in reviews:
        reviewed.provenance["changed"] = "after return"
    dataset.cases[0].input.parameters["n"] = 2
    assert report.dataset.cases[0].input.parameters == {"n": 1}
    assert report.dataset.cases[0].judgments.relevance == {"source": 1}
    for case in report.results:
        assert case.attempt.input.parameters == {"n": 1}
        assert case.attempt.output is not None
        assert case.attempt.output.final.provenance == {"model": "first"}
        assert case.attempt.review is not None and case.attempt.review.provenance == {}
        assert case.assessment is not None and case.assessment.answer_correct is True


def test_attempts_bind_the_declared_system_configuration_on_success_and_failure() -> None:
    config = RAGEvalConfig(system={"model": "configured", "search": {"limit": 10}})

    def pipeline(request: RAGInput) -> RAGTrace:
        config.system["model"] = "caller mutation during execution"
        if request.id == "failed":
            raise RuntimeError("offline")
        return _trace(request)

    report = evaluate_rag(_dataset("success", "failed"), pipeline, config=config)
    assert report.config.system == {"model": "configured", "search": {"limit": 10}}
    for result in report.results:
        assert result.attempt.system == report.config.system
    report.results[0].attempt.system["search"] = "changed retained report"
    assert report.results[1].attempt.system == report.config.system


def test_all_dataset_and_config_validation_happens_before_callbacks() -> None:
    called = False

    def pipeline(request: RAGInput) -> RAGTrace:
        nonlocal called
        called = True
        return _trace(request)

    dataset = _dataset("first", "second")
    dataset.cases[1].judgments.relevance["source"] = -1
    with pytest.raises(ValueError):
        evaluate_rag(dataset, pipeline)
    with pytest.raises(ValueError, match="max_concurrency"):
        evaluate_rag(_dataset(), pipeline, config=RAGEvalConfig(max_concurrency=2))
    invalid = RAGEvalConfig.model_construct(k=0)
    with pytest.raises(ValueError):
        evaluate_rag(_dataset(), pipeline, config=invalid)
    assert called is False


def test_live_run_rejects_prebound_review_before_calling() -> None:
    dataset = _dataset()
    case = dataset.cases[0]
    reviewed = RAGAnswerReview(trace_sha256=_trace(case.input).sha256, correct=True)
    dataset.cases[0] = case.model_copy(update={"judgments": RAGJudgments(answer=reviewed)})

    def pipeline(request: RAGInput) -> RAGTrace:
        pytest.fail("prebound review must be rejected before execution")

    with pytest.raises(ValueError, match="prebound"):
        evaluate_rag(dataset, pipeline)


def test_failures_are_redacted_and_later_cases_still_run() -> None:
    calls: list[str] = []
    private = "private document and api key"
    long_error = type("E" * 200, (Exception,), {})

    def pipeline(request: RAGInput) -> RAGTrace:
        calls.append(request.id)
        if request.id == "failed":
            raise long_error(private)
        return _trace(request)

    report = evaluate_rag(_dataset("failed", "success"), pipeline)
    assert calls == ["failed", "success"]
    failed, success = [item.attempt for item in report.results]
    assert failed.error is not None
    assert failed.error.phase == "pipeline" and failed.error.type == "E" * 128
    assert failed.output is None and success.output is not None
    assert private not in report.model_dump_json()
    assert report.summary.pipeline_failures == 1
    assert report.summary.planned == 2 and report.summary.recorded == 2


@pytest.mark.parametrize("malformation", ["wrong_type", "wrong_id", "wrong_query", "bad_trace"])
def test_invalid_outputs_are_output_failures_without_calling_review(malformation: str) -> None:
    def pipeline(request: RAGInput) -> RAGTrace:
        if malformation == "wrong_type":
            return cast(RAGTrace, {"answer": "unvalidated"})
        result = _trace(request)
        if malformation == "wrong_id":
            return result.model_copy(update={"id": "different"})
        if malformation == "wrong_query":
            return result.model_copy(update={"query": "different"})
        if result.retrieval is not None:
            result.retrieval.ids.append("source")
        return result

    def review(request: RAGReviewInput) -> RAGReview:
        pytest.fail("malformed output must not reach a review callback")

    result = evaluate_rag(_dataset(), pipeline, review=review).results[0]
    assert result.attempt.error is not None and result.attempt.error.phase == "output"
    assert result.attempt.review_duration_seconds is None


@pytest.mark.parametrize("problem", ["throws", "wrong_type", "stale", "no_answer", "no_context"])
def test_review_failure_preserves_valid_pipeline_observation(problem: str) -> None:
    def pipeline(request: RAGInput) -> RAGTrace:
        result = _trace(request)
        if problem == "no_answer":
            return result.model_copy(update={"answer": None})
        if problem == "no_context":
            return result.model_copy(update={"context": None})
        return result

    def review(request: RAGReviewInput) -> RAGReview:
        if problem == "throws":
            raise RuntimeError("private judge prompt")
        if problem == "wrong_type":
            return cast(RAGReview, True)
        digest = "0" * 64 if problem == "stale" else request.sha256
        return RAGReview(review_input_sha256=digest, correct=True, grounded=True)

    report = evaluate_rag(_dataset(), pipeline, review=review)
    attempt = report.results[0].attempt
    assert attempt.output is not None and attempt.error is None and attempt.review is None
    assert attempt.review_error is not None and attempt.review_error.phase == "review"
    assert attempt.review_duration_seconds is not None
    assert "private judge prompt" not in report.model_dump_json()
    assert report.summary.review_failures == 1


def test_custom_review_scores_do_not_require_a_generated_answer() -> None:
    def pipeline(request: RAGInput) -> RAGTrace:
        return RAGTrace(id=request.id, query=request.query, retrieval=RankedOutput(ids=[]))

    def review(request: RAGReviewInput) -> RAGReview:
        return RAGReview(
            review_input_sha256=request.sha256,
            scores={"retrieval_utility": RAGScore(value=None, reason="no relevant labels")},
        )

    attempt = evaluate_rag(_dataset(), pipeline, review=review).results[0].attempt
    assert attempt.review is not None and attempt.review_error is None


def test_repeats_retain_each_output_and_allow_explicit_model_routing() -> None:
    count = 0

    def pipeline(request: RAGInput) -> RAGOutput:
        nonlocal count
        count += 1
        final = _trace(request).model_copy(update={"answer": f"answer {count}"})
        final.provenance["model"] = f"routed model {count}"
        step = RAGTrace(id=f"{request.id}-rewrite", query="rewritten intermediate query")
        return RAGOutput(final=final, steps=[step])

    report = evaluate_rag(_dataset("first", "second"), pipeline, config=RAGEvalConfig(repeats=2))
    assert count == 4
    assert [(item.attempt.case_id, item.attempt.repeat) for item in report.results] == [
        ("first", 0),
        ("first", 1),
        ("second", 0),
        ("second", 1),
    ]
    for position, item in enumerate(report.results, start=1):
        output = item.attempt.output
        assert output is not None and output.final.answer == f"answer {position}"
        assert output.final.provenance == {"model": f"routed model {position}"}
        assert len(output.steps) == 1 and item.attempt.error is None
    assert report.summary.unstable_cases == ["first", "second"]


@pytest.mark.parametrize("changed", ["intermediate", "references", "parameters"])
def test_review_hash_covers_intermediate_steps_references_and_parameters(changed: str) -> None:
    def pipeline(request: RAGInput) -> RAGOutput:
        return RAGOutput(
            final=_trace(request), steps=[RAGTrace(id="rewrite", query="rewritten query")]
        )

    def review(request: RAGReviewInput) -> RAGReview:
        if changed == "intermediate":
            request.output.steps[0].provenance["changed"] = "source"
        elif changed == "references":
            request.case.judgments.relevance["source"] = 0
        else:
            request.case.input.parameters["n"] = 2
        return RAGReview(review_input_sha256=request.sha256, correct=True)

    attempt = evaluate_rag(_dataset(), pipeline, review=review).results[0].attempt
    assert attempt.output is not None and attempt.output.steps[0].provenance == {}
    assert attempt.review_error is not None and attempt.review_error.type == "ValueError"


def test_sync_interrupt_propagates_and_evaluator_does_not_close_backend() -> None:
    class Backend:
        closed = False

        def __call__(self, request: RAGInput) -> RAGTrace:
            raise KeyboardInterrupt

        def close(self) -> None:
            self.closed = True

    backend = Backend()
    with pytest.raises(KeyboardInterrupt):
        evaluate_rag(_dataset(), backend)
    assert backend.closed is False


def test_callback_timing_excludes_review_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Four reads surround exactly the two callback invocations.
    times = iter([1.0, 3.0, 10.0, 17.0])
    monkeypatch.setattr("kayak.eval._rag_runner.perf_counter", lambda: next(times))

    def review(request: RAGReviewInput) -> RAGReview:
        return RAGReview(review_input_sha256=request.sha256, correct=True)

    attempt = evaluate_rag(_dataset(), _trace, review=review).results[0].attempt
    assert attempt.duration_seconds == 2.0 and attempt.review_duration_seconds == 7.0
    assert next(times, None) is None


def test_async_execution_is_bounded_and_retains_dataset_order() -> None:
    async def exercise() -> None:
        release_first = asyncio.Event()
        second_started = asyncio.Event()
        active = 0
        maximum = 0
        finish_order: list[str] = []

        async def pipeline(request: RAGInput) -> RAGTrace:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                if request.id == "first":
                    await release_first.wait()
                elif request.id == "second":
                    second_started.set()
                finish_order.append(request.id)
                return _trace(request)
            finally:
                active -= 1

        task = asyncio.create_task(
            aevaluate_rag(
                _dataset("first", "second", "third"),
                pipeline,
                config=RAGEvalConfig(max_concurrency=2),
            )
        )
        await second_started.wait()
        assert maximum == 2 and not task.done()
        release_first.set()
        report = await task
        assert active == 0 and maximum == 2
        assert finish_order == ["second", "third", "first"]
        assert [item.attempt.case_id for item in report.results] == ["first", "second", "third"]

    asyncio.run(exercise())


def test_async_cancellation_awaits_workers_and_does_not_start_queued_cases() -> None:
    async def exercise() -> None:
        both_started = asyncio.Event()
        starts: list[str] = []
        stopped: list[str] = []

        async def pipeline(request: RAGInput) -> RAGTrace:
            starts.append(request.id)
            if len(starts) == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
                return _trace(request)
            finally:
                await asyncio.sleep(0)
                stopped.append(request.id)

        task = asyncio.create_task(
            aevaluate_rag(
                _dataset("first", "second", "queued"),
                pipeline,
                config=RAGEvalConfig(max_concurrency=2),
            )
        )
        await both_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert starts == ["first", "second"]
        assert sorted(stopped) == ["first", "second"]
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(exercise())


def test_async_allocates_workers_instead_of_a_task_per_queued_case() -> None:
    async def exercise() -> None:
        workers_started = asyncio.Event()
        release = asyncio.Event()
        started = 0

        async def pipeline(request: RAGInput) -> RAGTrace:
            nonlocal started
            started += 1
            if started == 3:
                workers_started.set()
            await release.wait()
            return _trace(request)

        task = asyncio.create_task(
            aevaluate_rag(
                _dataset(*(f"case-{position}" for position in range(20))),
                pipeline,
                config=RAGEvalConfig(max_concurrency=3),
            )
        )
        await workers_started.wait()
        # This exercise, the evaluator, and exactly three workers own tasks.
        assert len(asyncio.all_tasks()) == 5
        assert started == 3
        release.set()
        report = await task
        assert report.summary.recorded == 20

    asyncio.run(exercise())


def test_callback_cancellation_cancels_other_workers() -> None:
    async def exercise() -> None:
        second_started = asyncio.Event()
        second_stopped = asyncio.Event()

        async def pipeline(request: RAGInput) -> RAGTrace:
            if request.id == "first":
                await second_started.wait()
                raise asyncio.CancelledError
            second_started.set()
            try:
                await asyncio.Event().wait()
                return _trace(request)
            finally:
                second_stopped.set()

        with pytest.raises(asyncio.CancelledError):
            await aevaluate_rag(
                _dataset("first", "second"), pipeline, config=RAGEvalConfig(max_concurrency=2)
            )
        assert second_stopped.is_set()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(exercise())


def test_async_base_exception_propagates_after_other_workers_stop() -> None:
    class StopRun(BaseException):
        pass

    async def exercise() -> None:
        second_started = asyncio.Event()
        second_stopped = asyncio.Event()

        async def pipeline(request: RAGInput) -> RAGTrace:
            if request.id == "first":
                await second_started.wait()
                raise StopRun
            second_started.set()
            try:
                await asyncio.Event().wait()
                return _trace(request)
            finally:
                second_stopped.set()

        with pytest.raises(StopRun):
            await aevaluate_rag(
                _dataset("first", "second"), pipeline, config=RAGEvalConfig(max_concurrency=2)
            )
        assert second_stopped.is_set()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(exercise())


def test_async_and_sync_assessments_match_including_review_failures() -> None:
    def pipeline(request: RAGInput) -> RAGTrace:
        if request.id == "pipeline_failure":
            raise RuntimeError("unavailable")
        return _trace(request)

    def review(request: RAGReviewInput) -> RAGReview:
        if request.case.input.id == "review_failure":
            raise ValueError("no judgment")
        return RAGReview(review_input_sha256=request.sha256, correct=False, grounded=True)

    async def apipeline(request: RAGInput) -> RAGTrace:
        return pipeline(request)

    async def areview(request: RAGReviewInput) -> RAGReview:
        return review(request)

    dataset = _dataset("success", "pipeline_failure", "review_failure")
    synchronous = evaluate_rag(dataset, pipeline, review=review)
    asynchronous = asyncio.run(aevaluate_rag(dataset, apipeline, review=areview))
    assert [item.assessment for item in synchronous.results] == [
        item.assessment for item in asynchronous.results
    ]
    assert synchronous.summary.pipeline_failures == asynchronous.summary.pipeline_failures == 1
    assert synchronous.summary.review_failures == asynchronous.summary.review_failures == 1
    for first, second in zip(synchronous.results, asynchronous.results, strict=True):
        assert first.attempt.model_dump(
            exclude={"duration_seconds", "review_duration_seconds"}
        ) == second.attempt.model_dump(exclude={"duration_seconds", "review_duration_seconds"})


def test_async_invalid_config_does_not_execute_callbacks() -> None:
    called = False

    async def pipeline(request: RAGInput) -> RAGTrace:
        nonlocal called
        called = True
        return _trace(request)

    with pytest.raises(ValueError):
        asyncio.run(aevaluate_rag(_dataset(), pipeline, config=RAGEvalConfig.model_construct(k=0)))
    assert called is False


def test_completion_hook_retains_progress_before_later_interruption() -> None:
    retained: list[RAGAttempt] = []

    def pipeline(request: RAGInput) -> RAGTrace:
        if request.id == "interrupt":
            raise KeyboardInterrupt
        if request.id == "failure":
            raise RuntimeError("backend unavailable")
        return _trace(request)

    with pytest.raises(KeyboardInterrupt):
        evaluate_rag(
            _dataset("success", "failure", "interrupt"), pipeline, on_attempt=retained.append
        )
    assert [attempt.case_id for attempt in retained] == ["success", "failure"]
    assert retained[0].output is not None
    assert retained[1].error is not None and retained[1].error.phase == "pipeline"


def test_completion_hook_owns_a_snapshot_and_does_not_extend_callback_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([2.0, 5.0])
    monkeypatch.setattr("kayak.eval._rag_runner.perf_counter", lambda: next(times))

    def record(attempt: RAGAttempt) -> None:
        attempt.input.parameters.clear()
        assert attempt.output is not None
        attempt.output.final.provenance.clear()

    report = evaluate_rag(_dataset(), _trace, on_attempt=record)
    attempt = report.results[0].attempt
    assert attempt.input.parameters == {"n": 1}
    assert attempt.output is not None and attempt.output.final.provenance == {"model": "first"}
    assert attempt.duration_seconds == 3.0


def test_completion_hook_failure_propagates_without_starting_later_cases() -> None:
    called: list[str] = []

    def pipeline(request: RAGInput) -> RAGTrace:
        called.append(request.id)
        return _trace(request)

    def record(attempt: RAGAttempt) -> None:
        raise OSError("journal unavailable")

    with pytest.raises(OSError, match="journal unavailable"):
        evaluate_rag(_dataset("first", "queued"), pipeline, on_attempt=record)
    assert called == ["first"]


def test_async_completion_hook_failure_stops_and_awaits_other_workers() -> None:
    async def exercise() -> None:
        second_started = asyncio.Event()
        second_stopped = asyncio.Event()
        retained: list[RAGAttempt] = []

        async def pipeline(request: RAGInput) -> RAGTrace:
            if request.id == "first":
                return _trace(request)
            second_started.set()
            try:
                await asyncio.Event().wait()
                return _trace(request)
            finally:
                second_stopped.set()

        async def record(attempt: RAGAttempt) -> None:
            retained.append(attempt)
            await second_started.wait()
            raise OSError("journal unavailable")

        with pytest.raises(OSError, match="journal unavailable"):
            await aevaluate_rag(
                _dataset("first", "second", "queued"),
                pipeline,
                config=RAGEvalConfig(max_concurrency=2),
                on_attempt=record,
            )
        assert [attempt.case_id for attempt in retained] == ["first"]
        assert second_stopped.is_set()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(exercise())


def test_async_hook_retains_completion_and_is_detached_from_the_report() -> None:
    retained: list[RAGAttempt] = []

    async def pipeline(request: RAGInput) -> RAGTrace:
        return _trace(request)

    async def record(attempt: RAGAttempt) -> None:
        retained.append(attempt)
        attempt.input.parameters.clear()

    report = asyncio.run(aevaluate_rag(_dataset(), pipeline, on_attempt=record))
    assert len(retained) == 1
    assert retained[0].input.parameters == {}
    assert report.results[0].attempt.input.parameters == {"n": 1}
