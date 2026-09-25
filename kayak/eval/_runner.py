"""Run real decision calls, retain failures, and checkpoint reviewable evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal
from typing import Protocol as BackendProtocol

from .. import decisions
from ..client import Client
from ..decisions import Choice, DecisionRequest, DecisionResult, ModelInfo
from ..errors import InferenceError, KayakError
from ..runtime import Model
from ._measurement import environment, memory_snapshot, synchronize
from ._metrics import summarize
from ._schema import (
    Attempt,
    Example,
    Failure,
    Observation,
    PredictionArtifact,
    Protocol,
    Report,
    Suite,
)


class DecisionBackend(BackendProtocol):
    """Models, clients, and adapters share the existing typed decision interface."""

    def decide(
        self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult: ...


def _attempt(
    backend: DecisionBackend,
    question: Choice,
    example: Example,
    expected_model: ModelInfo | None,
    sync: Callable[[], None],
) -> Attempt:
    request = DecisionRequest(state=example.text, questions={"intent": question})
    questions = request.model_copy(deep=True).questions
    seconds = 0.0
    try:
        sync()
        start = perf_counter()
        call_failed = False
        try:
            try:
                result = backend.decide(state=request.state, questions=questions)
            except BaseException:
                call_failed = True
                raise
            finally:
                try:
                    sync()
                except Exception:
                    # A cleanup failure must not erase the original call failure.
                    if not call_failed:
                        raise
        finally:
            seconds = perf_counter() - start
        result = DecisionResult.model_validate(result)
        decisions.check_result(request, result)
        if expected_model is not None and result.model != expected_model:
            raise InferenceError("model identity changed during evaluation")
        return Attempt(seconds=seconds, result=result)
    except Exception as exc:
        # Do not copy arbitrary exception messages, which can contain input or credentials.
        return Attempt(
            seconds=seconds,
            error=Failure(
                type=type(exc).__name__,
                request_id=exc.request_id if isinstance(exc, KayakError) else None,
            ),
        )


def _checkpoint(report: Report, output: Path) -> None:
    report.summary = summarize(report)
    payload = report.model_dump(mode="python", exclude={"observations"})
    temporary = output / "report.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output / "report.json")


def evaluate(
    backend: DecisionBackend,
    suite: Suite,
    *,
    output: str | Path,
    warmups: int = 1,
    repeats: int = 1,
    seed: int = 42,
    sync: Callable[[], None] | None = None,
    max_memory_gib: float | None = None,
    config: dict[str, object] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Report:
    """Evaluate first-call quality and repeated-call latency; never close the backend.

    Warmups use the first shuffled example and are excluded from measured calls.
    Quality uses the first measured attempt per example, including failures.
    `sync` is for custom asynchronous adapters; Model GPU calls synchronize
    automatically. Model loading is the caller's responsibility, outside timing.
    The output directory must not exist. Raw predictions are flushed per example.
    """
    protocol = Protocol(warmups=warmups, repeats=repeats, seed=seed)
    suite = Suite.model_validate(suite.model_dump())  # Revalidate and snapshot mutable containers.
    if max_memory_gib is not None and (not math.isfinite(max_memory_gib) or max_memory_gib <= 0):
        raise ValueError("max_memory_gib must be finite and positive")
    if max_memory_gib is not None and not isinstance(backend, Model):
        raise ValueError("memory budgets require a local Model with a known execution device")
    device = backend.info.device if isinstance(backend, Model) else None

    def synchronize_call() -> None:
        if sync is not None:
            sync()
        else:
            synchronize(device)

    transport: Literal["local", "http", "custom"]
    if isinstance(backend, Model):
        transport = "local"
    elif isinstance(backend, Client):
        transport = "http"
    else:
        transport = "custom"
    prediction_hash = hashlib.sha256()
    prediction_bytes = 0
    recorded_environment = environment(backend if isinstance(backend, Model) else None)
    if sync is not None:
        recorded_environment["synchronization"] = "custom"
    report = Report(
        schema_version=3,
        created_at=datetime.now(UTC).isoformat(),
        suite=suite,
        suite_sha256=suite.sha256,
        protocol=protocol,
        transport=transport,
        environment=recorded_environment,
        config={**(config or {}), "max_memory_gib": max_memory_gib},
        prediction_artifact=PredictionArtifact(sha256=prediction_hash.hexdigest(), bytes=0),
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    order = list(suite.examples)
    random.Random(seed).shuffle(order)

    def memory_exceeded() -> bool:
        for key, value in memory_snapshot(device).items():
            if value is not None:
                report.memory[key] = max(report.memory.get(key) or 0, value)
        if max_memory_gib is None:
            return False
        if device and device.startswith("mps"):
            key = "mps_driver_snapshot_bytes"
        elif device and device.startswith("cuda"):
            key = "cuda_peak_tensor_bytes"
        else:
            key = "process_peak_rss_bytes"
        measured = report.memory.get(key)
        if measured is None:
            raise ValueError(f"memory budget cannot be enforced: {key} is unavailable")
        return measured > max_memory_gib * 1024**3

    try:
        with (destination / "predictions.jsonl").open("xb") as stream:
            _checkpoint(report, destination)
            if memory_exceeded():
                report.error = "MemoryBudgetExceeded"
            else:
                for _ in range(warmups):
                    attempt = _attempt(
                        backend, suite.question, order[0], report.model, synchronize_call
                    )
                    report.warmups.append(attempt)
                    if attempt.result is not None:
                        report.model = attempt.result.model
                    if memory_exceeded():
                        report.error = "MemoryBudgetExceeded"
                        break
                for example in order:
                    if report.error is not None:
                        break
                    attempts: list[Attempt] = []
                    try:
                        for _ in range(repeats):
                            attempt = _attempt(
                                backend, suite.question, example, report.model, synchronize_call
                            )
                            attempts.append(attempt)
                            if attempt.result is not None:
                                report.model = attempt.result.model
                            if memory_exceeded():
                                report.error = "MemoryBudgetExceeded"
                                break
                    finally:
                        if attempts:
                            row = Observation(id=example.id, attempts=attempts)
                            raw = (row.model_dump_json() + "\n").encode("utf-8")
                            stream.write(raw)
                            stream.flush()
                            prediction_hash.update(raw)
                            prediction_bytes += len(raw)
                            report.prediction_artifact = PredictionArtifact(
                                sha256=prediction_hash.hexdigest(), bytes=prediction_bytes
                            )
                            report.observations.append(row)
                    if progress:
                        progress(len(report.observations), len(order))
                    if len(report.observations) % 25 == 0:
                        _checkpoint(report, destination)
        failed = any(attempt.error is not None for attempt in report.warmups)
        for observation in report.observations:
            if any(attempt.error is not None for attempt in observation.attempts):
                failed = True
        report.status = "failed" if failed or report.error else "complete"
    except BaseException as exc:
        report.status = "interrupted"
        report.error = type(exc).__name__
        raise
    finally:
        _checkpoint(report, destination)
    return report


def load_report(path: str | Path) -> Report:
    """Verify saved observations; version 1 remains legacy without byte integrity."""
    path = Path(path)
    path = path / "report.json" if path.is_dir() else path
    report = Report.model_validate_json(path.read_bytes())
    if report.observations:
        raise ValueError("saved reports store observations only in predictions.jsonl")
    predictions = path.parent / "predictions.jsonl"
    raw = predictions.read_bytes()
    artifact = report.prediction_artifact
    if artifact is not None:
        if artifact.bytes > len(raw) or (report.status != "running" and artifact.bytes != len(raw)):
            raise ValueError("prediction artifact byte count does not match the saved file")
        committed = raw[: artifact.bytes]
        if committed and not committed.endswith(b"\n"):
            raise ValueError("prediction checkpoint ends inside a row")
        if hashlib.sha256(committed).hexdigest() != artifact.sha256:
            raise ValueError("prediction artifact hash does not match the saved bytes")
        if raw and not raw.endswith(b"\n"):
            raise ValueError("prediction file has a partial final row")
    report.observations = [Observation.model_validate_json(line) for line in raw.splitlines()]
    # A killed process can leave predictions newer than its last metadata checkpoint.
    if report.status == "running" and report.model is None:
        for row in report.observations:
            for attempt in row.attempts:
                if attempt.result is not None:
                    report.model = attempt.result.model
                    break
            if report.model is not None:
                break
    examples = {example.id: example for example in report.suite.examples}
    expected_order = list(examples)
    random.Random(report.protocol.seed).shuffle(expected_order)
    observed_order = [row.id for row in report.observations]
    if observed_order != expected_order[: len(observed_order)]:
        raise ValueError("prediction order differs from the declared execution order")
    warmup_request = DecisionRequest(
        state=examples[expected_order[0]].text, questions={"intent": report.suite.question}
    )
    if len(report.warmups) > report.protocol.warmups:
        raise ValueError("too many warmup attempts")
    if (
        report.status != "running"
        and report.observations
        and len(report.warmups) != report.protocol.warmups
    ):
        raise ValueError("prediction calls started before the declared warmups finished")
    for attempt in report.warmups:
        if attempt.result is not None:
            decisions.check_result(warmup_request, attempt.result)
            if attempt.result.model != report.model:
                raise ValueError("warmup model differs from report model")
    seen: set[str] = set()
    for index, row in enumerate(report.observations):
        if row.id in seen or row.id not in examples:
            raise ValueError("duplicate or unknown prediction ID")
        seen.add(row.id)
        if len(row.attempts) > report.protocol.repeats:
            raise ValueError("too many prediction attempts")
        if report.status == "complete" and len(row.attempts) != report.protocol.repeats:
            raise ValueError("completed run has missing attempts")
        if index < len(report.observations) - 1 and len(row.attempts) != report.protocol.repeats:
            raise ValueError("only the final prediction row may have missing attempts")
        request = DecisionRequest(
            state=examples[row.id].text, questions={"intent": report.suite.question}
        )
        for attempt in row.attempts:
            if attempt.result:
                decisions.check_result(request, attempt.result)
                if attempt.result.model != report.model:
                    raise ValueError("prediction model differs from report model")
            elif report.status == "complete":
                raise ValueError("completed run contains failed attempts")
    if report.status == "complete" and (
        len(seen) != len(examples)
        or report.error is not None
        or report.model is None
        or len(report.warmups) != report.protocol.warmups
        or any(attempt.error for attempt in report.warmups)
    ):
        raise ValueError("completed run has missing cases, failed warmups or a run error")
    summary = summarize(report, legacy=report.schema_version < 3)
    if report.status == "failed" and not (
        report.error or summary["failed_calls"] or summary["failed_warmups"]
    ):
        raise ValueError("failed run has no recorded failure")
    if report.status == "interrupted" and not report.error:
        raise ValueError("interrupted run has no recorded interruption")
    if (
        report.status != "running"
        and not report.error
        and (
            len(report.warmups) != report.protocol.warmups
            or len(report.observations) != len(examples)
            or any(len(row.attempts) != report.protocol.repeats for row in report.observations)
        )
    ):
        raise ValueError("unfinished terminal run has no recorded run error")
    if report.status != "running" and summary != report.summary:
        raise ValueError("summary does not match saved predictions")
    report.summary = summary
    return report
