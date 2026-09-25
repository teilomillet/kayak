"""Fixed-denominator quality checks and conservative local experiment comparisons."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kayak.decisions import DecisionRequest, DecisionResult, check_result
from kayak.errors import InferenceError

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Case(Record):
    id: str = Field(min_length=1)
    split: Literal["dev", "holdout"]
    tags: list[str]
    request: DecisionRequest
    expected: dict[str, str]

    @model_validator(mode="after")
    def valid_labels(self) -> Self:
        if set(self.expected) != set(self.request.questions):
            raise ValueError("label every question exactly once")
        for question, answer in self.expected.items():
            if answer not in self.request.questions[question].criteria:
                raise ValueError("expected answer must be a candidate ID")
        return self


class Sample(Record):
    seconds: Positive
    result: DecisionResult | None = None
    error: str | None = None

    @model_validator(mode="after")
    def one_outcome(self) -> Self:
        if (self.result is None) == (self.error is None):
            raise ValueError("sample must contain exactly one result or error")
        return self


class CaseRun(Record):
    case: Case
    warmups: list[Sample] = Field(default_factory=list)
    samples: list[Sample] = Field(default_factory=list)
    memory: dict[str, int | None] = Field(default_factory=dict)


class Protocol(Record):
    transport: Literal["local", "http"] = "local"
    split: Literal["dev", "holdout", "all"] = "all"
    warmups: int = Field(default=2, ge=1)
    repeats: int = Field(default=10, ge=2)
    seed: int = 20260924


class Run(Record):
    schema_version: Literal[1] = 1
    run_id: str
    created_at: str
    status: Literal["running", "passed", "failed"] = "running"
    protocol: Protocol
    suite_sha256: str
    harness_sha256: str
    source: dict[str, object]
    environment: dict[str, object]
    config: dict[str, object]
    model: dict[str, str] = Field(default_factory=dict)
    load_seconds: float | None = None
    cases: list[CaseRun]
    memory_after_load: dict[str, int | None] = Field(default_factory=dict)
    system_before: dict[str, object] = Field(default_factory=dict)
    system_after: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


def read_cases(path: Path, split: str) -> tuple[list[Case], str]:
    raw = path.read_bytes()
    cases = [Case.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("suite must have nonempty, unique case IDs")
    selected = [case for case in cases if split == "all" or case.split == split]
    if not selected:
        raise ValueError(f"suite has no {split} cases")
    return selected, hashlib.sha256(raw).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None, "cv": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        # Avoid representing a handful of samples as a credible tail estimate.
        "p95": percentile(values, 0.95) if len(values) >= 20 else None,
        "min": min(values),
        "max": max(values),
        "cv": statistics.pstdev(values) / statistics.mean(values),
    }


def result_drift(reference: DecisionResult, actual: DecisionResult) -> tuple[int, float]:
    if reference.model != actual.model or reference.input_tokens != actual.input_tokens:
        raise ValueError("model identity, precision, or token accounting changed")
    if list(reference.answers) != list(actual.answers):
        raise ValueError("question IDs/order changed")
    disagreements, error = 0, 0.0
    for key, answer in reference.answers.items():
        other = actual.answers[key]
        if list(answer.scores) != list(other.scores):
            raise ValueError("candidate IDs/order changed")
        disagreements += answer.choice != other.choice
        error = max(error, *(abs(value - other.scores[k]) for k, value in answer.scores.items()))
    return disagreements, error


def summarize(run: Run) -> dict[str, object]:
    cases: dict[str, object] = {}
    correct, labels, failures = 0, 0, 0
    for entry in run.cases:
        # Missing and failed observations never shrink the quality denominator.
        expected = run.protocol.repeats * len(entry.case.expected)
        labels += expected
        case_correct = 0
        good = [sample for sample in entry.samples if sample.result is not None]
        failures += run.protocol.repeats - len(good)
        for sample in good:
            assert sample.result is not None
            case_correct += sum(
                sample.result.answers[key].choice == value
                for key, value in entry.case.expected.items()
            )
        correct += case_correct
        seconds = [sample.seconds for sample in good]
        tokens = sum(sample.result.input_tokens for sample in good if sample.result is not None)
        cases[entry.case.id] = {
            "split": entry.case.split,
            "tags": entry.case.tags,
            "question_count": len(entry.case.request.questions),
            "candidate_count": sum(len(q.criteria) for q in entry.case.request.questions.values()),
            "latency_seconds": distribution(seconds),
            "first_observed_seconds": entry.warmups[0].seconds if entry.warmups else None,
            "correct": case_correct,
            "labels": expected,
            "request_per_second": len(good) / sum(seconds) if seconds else None,
            "input_tokens_per_second": tokens / sum(seconds) if seconds else None,
            "memory": entry.memory,
        }
    return {
        "status": run.status,
        "correct": correct,
        "labels": labels,
        "accuracy": correct / labels if labels else 0,
        "failed_or_missing": failures,
        "cases": cases,
    }


def observations(run: Run) -> dict[str, list[DecisionResult]]:
    if run.status != "passed" or run.error:
        raise ValueError(f"run {run.run_id} did not pass")
    if not run.cases or len({entry.case.id for entry in run.cases}) != len(run.cases):
        raise ValueError("missing/duplicate cases")
    results: dict[str, list[DecisionResult]] = {}
    for entry in run.cases:
        if len(entry.samples) != run.protocol.repeats or len(entry.warmups) != run.protocol.warmups:
            raise ValueError("missing or extra measured/warmup observations")
        results[entry.case.id] = []
        for sample in entry.warmups + entry.samples:
            if sample.result is None or sample.error:
                raise ValueError("failed inference observation")
            check_result(entry.case.request, sample.result)
            if sample.result.model.model_dump() != run.model:
                raise ValueError("reported model differs from result identity")
            results[entry.case.id].append(sample.result)
    return results


def memory_high_water(run: Run) -> int:
    """One backend-specific counter; MPS is a maximum of snapshots, not a peak."""
    backend = run.model.get("device", "cpu").split(":")[0]
    key = {"mps": "mps_current_driver_bytes", "cuda": "cuda_peak_tensor_bytes"}.get(
        backend, "process_peak_rss_bytes"
    )
    counters = [run.memory_after_load, *(entry.memory for entry in run.cases)]
    values = [counter[key] for counter in counters if counter.get(key) is not None]
    if not values:
        raise ValueError(f"missing memory counter: {key}")
    return max(value for value in values if value is not None)


def compare(
    baseline: list[Run],
    candidate: list[Run],
    *,
    score_atol: float = 0.0,
    min_speedup: float = 0.05,
    max_case_regression: float = 0.10,
    max_memory_regression: float = 0.10,
) -> dict[str, object]:
    """Gate every result before comparing equal-weight per-case process medians.

    The bootstrap resamples whole process trials, never individual requests as
    independent experiments. Three trials is a floor, not a high-confidence study.
    """
    limits = (score_atol, min_speedup, max_case_regression, max_memory_regression)
    if any(not math.isfinite(value) or value < 0 for value in limits):
        raise ValueError("comparison thresholds must be finite and nonnegative")
    reasons: list[str] = []
    output: dict[str, object] = {
        "status": "rejected",
        "reasons": reasons,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thresholds": dict(
            zip(
                ("score_atol", "min_speedup", "max_case_regression", "max_memory_regression"),
                limits,
                strict=True,
            )
        ),
    }
    if not baseline or not candidate:
        reasons.append("both groups need runs")
        return output
    try:
        all_runs = baseline + candidate
        if len({run.run_id for run in all_runs}) != len(all_runs):
            raise ValueError("duplicate runs are not independent trials")
        first = baseline[0]
        expected_cases = [entry.case for entry in first.cases]
        for run in all_runs:
            for field in ("protocol", "suite_sha256", "harness_sha256", "environment", "model"):
                if getattr(run, field) != getattr(first, field):
                    raise ValueError(f"incompatible {field}")
            if [entry.case for entry in run.cases] != expected_cases:
                raise ValueError("case coverage/order/labels changed")
        for group in (baseline, candidate):
            if any(run.config != group[0].config or run.source != group[0].source for run in group):
                raise ValueError("configuration/source changed within a trial group")
        output["quality_accuracy"] = {
            "baseline_trials": [summarize(run)["accuracy"] for run in baseline],
            "candidate_trials": [summarize(run)["accuracy"] for run in candidate],
            "distinct_labeled_questions": sum(len(case.expected) for case in expected_cases),
        }
        reference = observations(first)
        max_error, disagreements = 0.0, 0
        case_drift = {key: 0.0 for key in reference}
        for run in all_runs:
            for key, results in observations(run).items():
                for actual in results:
                    changed, error = result_drift(reference[key][0], actual)
                    disagreements += changed
                    max_error = max(max_error, error)
                    case_drift[key] = max(case_drift[key], error)
        output.update(
            choice_disagreements=disagreements,
            max_abs_score_error=max_error,
            case_max_abs_score_error=case_drift,
        )
        if disagreements:
            reasons.append("selected answers changed (including within baseline trials)")
        if max_error > score_atol:
            reasons.append("score drift exceeds the declared absolute tolerance")
        before_memory = max(memory_high_water(run) for run in baseline)
        after_memory = max(memory_high_water(run) for run in candidate)
        output.update(baseline_memory_bytes=before_memory, candidate_memory_bytes=after_memory)
        if after_memory > before_memory * (1 + max_memory_regression):
            reasons.append("memory observation exceeds the allowed regression")
        medians = [
            [
                [
                    statistics.median(sample.seconds for sample in entry.samples)
                    for entry in run.cases
                ]
                for run in group
            ]
            for group in (baseline, candidate)
        ]
        before, after = medians
        case_speedups = {
            case.id: statistics.median(row[i] for row in before)
            / statistics.median(row[i] for row in after)
            for i, case in enumerate(expected_cases)
        }
        output["case_speedups"] = case_speedups
        speedup = statistics.geometric_mean(case_speedups.values())
        output["geomean_speedup"] = speedup
        if speedup < 1 + min_speedup:
            reasons.append("aggregate speedup does not clear the minimum improvement")
        if any(speedup < 1 / (1 + max_case_regression) for speedup in case_speedups.values()):
            reasons.append("at least one case exceeds the allowed latency regression")
        # Resample entire processes, preserving cross-case correlations. Recompute
        # the SAME statistic as the point estimate; a different aggregation can
        # produce an interval that admits an apparent win under skewed timings.
        rng = random.Random(20260924)
        bootstrap = []
        for _ in range(2000):
            before_sample = rng.choices(before, k=len(before))
            after_sample = rng.choices(after, k=len(after))
            bootstrap.append(
                statistics.geometric_mean(
                    statistics.median(row[i] for row in before_sample)
                    / statistics.median(row[i] for row in after_sample)
                    for i in range(len(expected_cases))
                )
            )
        lower, upper = percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)
        output.update(
            speedup_95pct_interval=[lower, upper],
            baseline_trials=len(baseline),
            candidate_trials=len(candidate),
        )
        if min(len(baseline), len(candidate)) < 3:
            reasons.append("fewer than three fresh processes per configuration")
        if lower < 1 + min_speedup:
            reasons.append("speedup interval does not clear the minimum improvement")
        output["status"] = "passed" if not reasons else "rejected"
    except (ValueError, KeyError, InferenceError) as exc:
        reasons.append(str(exc))
    return output
