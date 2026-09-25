"""Score portable RAG observations and retain verifiable, model-free reports."""

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal

from ._rag import RAGAnswerReview, RAGAssessment, assess_rag
from ._rag_experiment import (
    RAGAttempt,
    RAGCase,
    RAGCaseResult,
    RAGDataset,
    RAGEvalConfig,
    RAGGateResult,
    RAGMetricSummary,
    RAGOutput,
    RAGReport,
    RAGReviewInput,
    RAGSummary,
)
from ._schema import digest

_RANKING_METRICS = (
    "top1",
    "hit_at_k",
    "known_recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "judged_fraction_at_k",
)
_METRICS = (
    "routing.correct",
    *(
        f"{stage}.{metric}"
        for stage in ("retrieval", "reranking", "shortlist_reranking")
        for metric in _RANKING_METRICS
    ),
    "source_coverage.retrieval",
    "source_coverage.reranking_top_k",
    "source_coverage.context",
    "context.required_text_retained",
    "answer.correct",
    "answer.grounded",
    "pipeline.duration_seconds",
    "review.duration_seconds",
)


def _diagnostic(output: RAGOutput) -> bool:
    return output.final.replay is not None or any(step.replay is not None for step in output.steps)


def _matching_json(value: object) -> str:
    """Compare JSON values by type and list order, ignoring object key order.

    Python equality equates True with 1. Serialization preserves that distinction
    and does not alter the separate, order-sensitive artifact digest contract.
    """
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _assess(case: RAGCase, attempt: RAGAttempt, k: int) -> RAGAssessment | None:
    output = attempt.output
    if output is None:
        return None
    trace = output.final
    if trace.id != case.input.id or trace.query != case.input.query:
        raise ValueError("final trace ID and query must match the dataset input")
    judgments = case.judgments
    review = attempt.review
    if (
        review is not None
        and review.review_input_sha256 != RAGReviewInput(case=case, output=output).sha256
    ):
        raise ValueError("review belongs to different inputs, references, or outputs")
    if review is not None and (review.correct is not None or review.grounded is not None):
        original = case.judgments.answer
        if original is not None:
            # Validate the legacy binding before combining reviews; do not erase a stale one.
            assess_rag(trace, judgments, k=k)
            for prior, current in (
                (original.correct, review.correct),
                (original.grounded, review.grounded),
            ):
                if prior is not None and current is not None and prior != current:
                    raise ValueError("attempt review contradicts the dataset answer review")
        correct = review.correct
        grounded = review.grounded
        provenance = dict(review.provenance)
        if original is not None:
            correct = original.correct if correct is None else correct
            grounded = original.grounded if grounded is None else grounded
            provenance = {
                **{f"dataset.{key}": value for key, value in original.provenance.items()},
                **{f"attempt.{key}": value for key, value in review.provenance.items()},
            }
        judgments = case.judgments.model_copy(
            update={
                "answer": RAGAnswerReview(
                    trace_sha256=trace.sha256,
                    correct=correct,
                    grounded=grounded,
                    provenance=provenance,
                )
            }
        )
    assessment = assess_rag(trace, judgments, k=k)
    if _diagnostic(output):
        # Injected evidence in an earlier step also makes the final result diagnostic.
        assessment = assessment.model_copy(update={"ordinary_aggregate_eligible": False})
    return assessment


def _ordered_attempts(
    dataset: RAGDataset, attempts: Sequence[RAGAttempt], config: RAGEvalConfig
) -> list[RAGAttempt]:
    positions = {case.input.id: index for index, case in enumerate(dataset.cases)}
    inputs = {
        case.input.id: _matching_json(case.input.model_dump(mode="json")) for case in dataset.cases
    }
    system = _matching_json(config.system)
    seen: set[tuple[str, int]] = set()
    checked = []
    for attempt in attempts:
        attempt = RAGAttempt.model_validate(attempt.model_dump())
        if attempt.case_id not in positions:
            raise ValueError("attempt case ID is absent from the dataset")
        if _matching_json(attempt.input.model_dump(mode="json")) != inputs[attempt.case_id]:
            raise ValueError("attempt input differs from the complete dataset input")
        if _matching_json(attempt.system) != system:
            raise ValueError("attempt system settings differ from the configured system snapshot")
        if attempt.repeat >= config.repeats:
            raise ValueError("attempt repeat exceeds the configured repetitions")
        key = (attempt.case_id, attempt.repeat)
        if key in seen:
            raise ValueError("duplicate attempt for a case and repetition")
        seen.add(key)
        checked.append(attempt)
    return sorted(checked, key=lambda attempt: (positions[attempt.case_id], attempt.repeat))


def _values(result: RAGCaseResult) -> dict[str, float | None]:
    attempt = result.attempt
    values: dict[str, float | None] = {
        "pipeline.duration_seconds": attempt.duration_seconds,
        "review.duration_seconds": attempt.review_duration_seconds,
    }
    assessment = result.assessment
    if assessment is not None:
        values.update(
            {
                name: None if value is None else float(value)
                for name, value in (
                    ("routing.correct", assessment.route_correct),
                    ("context.required_text_retained", assessment.required_text_retained),
                    ("answer.correct", assessment.answer_correct),
                    ("answer.grounded", assessment.answer_grounded),
                )
            }
        )
        for stage, ranking in (
            ("retrieval", assessment.retrieval),
            ("reranking", assessment.reranking),
            ("shortlist_reranking", assessment.shortlist_reranking),
        ):
            if ranking is not None:
                values.update({f"{stage}.{name}": value for name, value in ranking.metrics.items()})
        values.update(
            {
                f"source_coverage.{name}": None if value is None else float(value)
                for name, value in assessment.source_coverage.items()
            }
        )
    if attempt.review is not None:
        values.update(
            {f"custom.{name}": score.value for name, score in attempt.review.scores.items()}
        )
    return values


def _metric_summary(values: list[float], eligible: int) -> RAGMetricSummary:
    return RAGMetricSummary(
        # statistics uses exact sums internally, avoiding overflow from finite native scores.
        mean=float(mean(values)) if values else None,
        observed=len(values),
        unknown=eligible - len(values),
        coverage=len(values) / eligible if eligible else 0.0,
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
        stddev=pstdev(values) if values else None,
    )


def _summarize(
    dataset: RAGDataset, results: list[RAGCaseResult], config: RAGEvalConfig
) -> RAGSummary:
    planned = len(dataset.cases) * config.repeats
    diagnostic = sum(
        result.attempt.output is not None and _diagnostic(result.attempt.output)
        for result in results
    )
    names = dict.fromkeys(_METRICS)
    rows: list[dict[str, float | None]] = []
    stage_errors: dict[str, int] = {}
    output_hashes: dict[str, set[str]] = {}
    for result in results:
        attempt = result.attempt
        values = _values(result)
        names.update(dict.fromkeys(values))
        output = attempt.output
        if output is not None:
            for trace in (*output.steps, output.final):
                for stage in trace.errors:
                    stage_errors[stage] = stage_errors.get(stage, 0) + 1
            if _diagnostic(output):
                continue
            output_hashes.setdefault(attempt.case_id, set()).add(output.sha256)
        rows.append(values)
    names.update(dict.fromkeys(gate.metric for gate in config.gates))
    metrics = {
        name: _metric_summary(
            [value for row in rows if (value := row.get(name)) is not None],
            planned - diagnostic,
        )
        for name in names
    }
    return RAGSummary(
        planned=planned,
        recorded=len(results),
        missing=planned - len(results),
        pipeline_failures=sum(result.attempt.error is not None for result in results),
        review_failures=sum(result.attempt.review_error is not None for result in results),
        stage_errors=stage_errors,
        diagnostic_excluded=diagnostic,
        metrics=metrics,
        unstable_cases=[
            case.input.id
            for case in dataset.cases
            if len(output_hashes.get(case.input.id, set())) > 1
        ],
    )


def _gates(config: RAGEvalConfig, summary: RAGSummary) -> list[RAGGateResult]:
    results = []
    for gate in config.gates:
        metric = summary.metrics[gate.metric]
        if metric.mean is None:
            results.append(
                RAGGateResult(metric=gate.metric, status="unknown", reason="no known values")
            )
        elif metric.coverage < gate.min_coverage:
            results.append(
                RAGGateResult(
                    metric=gate.metric,
                    status="unknown",
                    reason=f"coverage {metric.coverage:g} is below required {gate.min_coverage:g}",
                )
            )
        elif (gate.minimum is not None and metric.mean < gate.minimum) or (
            gate.maximum is not None and metric.mean > gate.maximum
        ):
            results.append(
                RAGGateResult(
                    metric=gate.metric, status="failed", reason="known-value mean is outside bounds"
                )
            )
        else:
            results.append(
                RAGGateResult(
                    metric=gate.metric,
                    status="passed",
                    reason="known-value mean and observation coverage satisfy the gate",
                )
            )
    return results


def score_rag(
    dataset: RAGDataset,
    attempts: Sequence[RAGAttempt],
    *,
    config: RAGEvalConfig | None = None,
) -> RAGReport:
    """Purely validate and score recorded attempts, including missing cases and failures.

    Attempts are ordered by dataset case and zero-based repetition. Only final
    traces receive final-task judgments; intermediate steps retain their own
    query and sources. Any declared diagnostic step excludes the whole attempt
    from ordinary metric aggregates and prevents experiment-level gate passage.
    Failures and missing attempts remain in metric coverage denominators without
    inventing quality values. Input and system matching preserves JSON scalar
    types and list order while ignoring object key order.
    """
    dataset = RAGDataset.model_validate(dataset.model_dump())
    config = RAGEvalConfig.model_validate((config or RAGEvalConfig()).model_dump())
    ordered = _ordered_attempts(dataset, attempts, config)
    cases = {case.input.id: case for case in dataset.cases}
    results = [
        RAGCaseResult(
            attempt=attempt, assessment=_assess(cases[attempt.case_id], attempt, config.k)
        )
        for attempt in ordered
    ]
    summary = _summarize(dataset, results, config)
    gates = _gates(config, summary)
    failed = bool(summary.pipeline_failures or summary.review_failures or summary.stage_errors)
    status: Literal["complete", "failed", "incomplete"]
    status = "incomplete" if summary.missing else "failed" if failed else "complete"
    return RAGReport(
        dataset=dataset,
        config=config,
        dataset_sha256=dataset.sha256,
        config_sha256=config.sha256,
        attempts_sha256=digest([attempt.model_dump(mode="json") for attempt in ordered]),
        status=status,
        passed=(
            status == "complete"
            and summary.diagnostic_excluded == 0
            and all(gate.status == "passed" for gate in gates)
        )
        if gates
        else None,
        results=results,
        summary=summary,
        gates=gates,
    )


def _checked_report(report: RAGReport) -> RAGReport:
    report = RAGReport.model_validate(report.model_dump())
    recalculated = score_rag(
        report.dataset, [result.attempt for result in report.results], config=report.config
    )
    if report.model_dump() != recalculated.model_dump():
        raise ValueError("RAG report hashes, assessments, or summary differ from retained records")
    return recalculated


def save_rag_report(report: RAGReport, path: str | Path) -> None:
    """Validate and atomically publish one JSON file without overwriting existing evidence."""
    checked = _checked_report(report)
    destination = Path(path)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(checked.model_dump_json(indent=2) + "\n")
        # Both paths share a filesystem. link is atomic and refuses an existing destination.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_rag_report(path: str | Path) -> RAGReport:
    """Read evidence and recompute bindings, assessments, summaries, and configured gates."""
    return _checked_report(RAGReport.model_validate_json(Path(path).read_bytes()))


def _number(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.6g}"


def render_rag_report(report: RAGReport) -> str:
    """Render checked evidence; durations and provenance remain caller declarations."""
    report = _checked_report(report)
    summary = report.summary
    gate_status = "not configured"
    if report.passed is not None:
        gate_status = "passed" if report.passed else "not passed"
    lines = [
        "# Kayak RAG evaluation report",
        "",
        f"**Execution status:** `{report.status}`. **Configured gates:** {gate_status}.",
        "",
        "Execution, timing, reviewer identity, and provenance are caller-declared observations.",
        "Consistency hashes detect altered records; they do not authenticate execution or a judge.",
        "Completion records all planned attempts; it does not establish answer correctness.",
        "",
        "| Observation | Attempts |",
        "| --- | ---: |",
        f"| Planned | {summary.planned} |",
        f"| Recorded | {summary.recorded} |",
        f"| Missing | {summary.missing} |",
        f"| Pipeline/output failures | {summary.pipeline_failures} |",
        f"| Review failures | {summary.review_failures} |",
        f"| Diagnostic attempts excluded from metrics | {summary.diagnostic_excluded} |",
        "",
        "| Metric | Known-value mean | Known | Unknown | Coverage | Min | Max | "
        "Population stddev |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in summary.metrics.items():
        lines.append(
            f"| `{name}` | {_number(metric.mean)} | {metric.observed} | {metric.unknown} | "
            f"{metric.coverage:.1%} | {_number(metric.minimum)} | {_number(metric.maximum)} | "
            f"{_number(metric.stddev)} |"
        )
    lines.extend(
        [
            "",
            "These are descriptive attempt statistics, not independent task samples",
            "or confidence intervals. Unknown quality is not zero or success. Pipeline",
            "failures and missing attempts remain in coverage denominators; diagnostic",
            "attempts are excluded. Native custom scores keep their declared scale.",
            "Earlier steps are retained, never scored against the final query's labels.",
            "Durations measure caller-observed callback wall time, not verified device time.",
            "",
        ]
    )
    if summary.stage_errors:
        lines.extend(
            ["| Stage | Reported errors across final and intermediate steps |", "| --- | ---: |"]
        )
        for stage, count in summary.stage_errors.items():
            lines.append(f"| `{stage}` | {count} |")
        lines.append("")
    if report.gates:
        lines.extend(["| Gate | Status | Reason |", "| --- | --- | --- |"])
        for gate in report.gates:
            lines.append(f"| `{gate.metric}` | {gate.status} | {gate.reason} |")
        lines.extend(
            [
                "",
                "Passing the experiment also requires every planned attempt, no diagnostic",
                "exclusions, and no pipeline, review, or stage errors. A passing metric",
                "on the ordinary subset cannot make a mixed diagnostic experiment pass.",
                "",
            ]
        )
    if summary.unstable_cases:
        lines.append(
            f"{len(summary.unstable_cases)} cases have different recorded outputs "
            "across eligible repeats."
        )
        lines.append(
            "This includes changed history or provenance and does not by itself identify a cause."
        )
        lines.append("")
    lines.extend(
        [
            f"Dataset SHA-256: `{report.dataset_sha256}`",
            "",
            f"Configuration SHA-256: `{report.config_sha256}`",
            "",
            f"Attempts SHA-256: `{report.attempts_sha256}`",
            "",
            "The JSON report retains exact inputs, labels, system configuration, attempts,",
            "and reviews. Its configuration snapshot does not configure an external backend.",
            "No model runs while loading, scoring, or rendering this report.",
            "",
        ]
    )
    return "\n".join(lines)
