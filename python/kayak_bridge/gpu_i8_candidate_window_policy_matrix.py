"""Summaries for GPU i8 candidate-window policy matrices.

This module owns aggregation across completed FastPlaid policy reports. It does
not run benchmarks, build indexes, or choose candidate windows at runtime.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from kayak_bridge.gpu_i8_candidate_window_policy import INPUT_CANDIDATE_K_POLICY


def summarize_policy_matrix(
    reports: Sequence[dict[str, Any]],
    *,
    baseline_policy: str = INPUT_CANDIDATE_K_POLICY,
) -> dict[str, Any]:
    """Return cross-policy recall and timing summaries.

    The baseline comparison is keyed by case source, case name, FastPlaid device,
    seed, and input candidate count. That keeps the comparison shape-explicit and
    avoids mixing rows with different vector counts.
    """

    policy_summaries = [_policy_summary(report) for report in reports]
    baseline_comparisons = _baseline_comparisons(
        reports,
        baseline_policy=baseline_policy,
    )
    blockers = [
        row
        for report in reports
        for row in _ok_rows(report)
        if _optional_float(
            row.get("gpu_resident_selected_recall_delta_vs_fastplaid")
        )
        is not None
        and float(row["gpu_resident_selected_recall_delta_vs_fastplaid"]) < 0.0
    ]
    return {
        "schema_version": 1,
        "baseline_candidate_window_policy": baseline_policy,
        "policy_count": len(reports),
        "policy_summaries": policy_summaries,
        "baseline_comparison_summary": _baseline_comparison_summary(
            baseline_comparisons
        ),
        "baseline_comparisons": baseline_comparisons,
        "resident_selected_negative_recall_delta_rows": [
            _row_identity(row) | {
                "input_candidate_k": row.get("input_candidate_k"),
                "effective_candidate_k": row.get("effective_candidate_k"),
                "gpu_resident_selected_recall_delta_vs_fastplaid": (
                    row.get("gpu_resident_selected_recall_delta_vs_fastplaid")
                ),
                "gpu_resident_selected_recall_at_k_vs_kayak_exact": (
                    row.get("gpu_resident_selected_recall_at_k_vs_kayak_exact")
                ),
                "fastplaid_recall_at_k_vs_kayak_exact": (
                    row.get("fastplaid_recall_at_k_vs_kayak_exact")
                ),
            }
            for row in blockers
        ],
    }


def _policy_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = _ok_rows(report)
    widened_rows = [
        row
        for row in rows
        if _candidate_k(row, "effective_candidate_k")
        > _candidate_k(row, "input_candidate_k")
    ]
    ratios = [
        _candidate_k(row, "effective_candidate_k")
        / float(_candidate_k(row, "input_candidate_k"))
        for row in rows
        if _candidate_k(row, "input_candidate_k") > 0
    ]
    resident_ratios = _floats(
        row.get(
            "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
        )
        for row in rows
    )
    resident_recall_deltas = _floats(
        row.get("gpu_resident_selected_recall_delta_vs_fastplaid")
        for row in rows
    )
    negative_recall_delta_rows = [
        delta for delta in resident_recall_deltas if delta < 0.0
    ]
    return {
        "case_source": _case_source(report),
        "candidate_window_policy": _candidate_window_policy(report),
        "status": report.get("status"),
        "row_count": len(report.get("rows", [])),
        "ok_row_count": len(rows),
        "negative_recall_delta_row_count": len(negative_recall_delta_rows),
        "widened_row_count": len(widened_rows),
        "mean_effective_to_input_candidate_k": _mean(ratios),
        "max_effective_to_input_candidate_k": max(ratios) if ratios else None,
        "min_gpu_resident_selected_recall_delta_vs_fastplaid": (
            min(resident_recall_deltas) if resident_recall_deltas else None
        ),
        "mean_gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second": (
            _mean(resident_ratios)
        ),
        "max_gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second": (
            max(resident_ratios) if resident_ratios else None
        ),
        "gpu_resident_selected_final_topk_position_agreement_min": _min_optional(
            row.get("gpu_resident_selected_final_topk_position_agreement")
            for row in rows
        ),
        "gpu_resident_selected_candidate_position_agreement_min": _min_optional(
            row.get("gpu_resident_selected_candidate_position_agreement_min")
            for row in rows
        ),
    }


def _baseline_comparisons(
    reports: Sequence[dict[str, Any]],
    *,
    baseline_policy: str,
) -> list[dict[str, Any]]:
    baseline_rows = {
        _baseline_key(report, row): row
        for report in reports
        if _candidate_window_policy(report) == baseline_policy
        for row in _ok_rows(report)
    }
    comparisons: list[dict[str, Any]] = []
    for report in reports:
        policy = _candidate_window_policy(report)
        if policy == baseline_policy:
            continue
        for row in _ok_rows(report):
            baseline = baseline_rows.get(_baseline_key(report, row))
            if baseline is None:
                continue
            comparisons.append(
                _row_identity(row)
                | {
                    "case_source": _case_source(report),
                    "candidate_window_policy": policy,
                    "baseline_candidate_window_policy": baseline_policy,
                    "input_candidate_k": row.get("input_candidate_k"),
                    "effective_candidate_k": row.get("effective_candidate_k"),
                    "baseline_effective_candidate_k": baseline.get(
                        "effective_candidate_k"
                    ),
                    "gpu_resident_selected_recall_delta_vs_baseline": (
                        _delta(
                            row.get(
                                "gpu_resident_selected_recall_at_k_vs_kayak_exact"
                            ),
                            baseline.get(
                                "gpu_resident_selected_recall_at_k_vs_kayak_exact"
                            ),
                        )
                    ),
                    "gpu_resident_selected_fastplaid_delta_delta_vs_baseline": (
                        _delta(
                            row.get(
                                "gpu_resident_selected_recall_delta_vs_fastplaid"
                            ),
                            baseline.get(
                                "gpu_resident_selected_recall_delta_vs_fastplaid"
                            ),
                        )
                    ),
                    "gpu_resident_selected_fastplaid_ratio_delta_vs_baseline": (
                        _delta(
                            row.get(
                                "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
                            ),
                            baseline.get(
                                "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
                            ),
                        )
                    ),
                    "gpu_resident_selected_fastplaid_ratio_vs_baseline": _ratio(
                        row.get(
                            "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
                        ),
                        baseline.get(
                            "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
                        ),
                    ),
                }
            )
    return comparisons


def _baseline_comparison_summary(
    comparisons: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    recall_deltas = _floats(
        row.get("gpu_resident_selected_recall_delta_vs_baseline")
        for row in comparisons
    )
    fastplaid_delta_deltas = _floats(
        row.get("gpu_resident_selected_fastplaid_delta_delta_vs_baseline")
        for row in comparisons
    )
    ratio_deltas = _floats(
        row.get("gpu_resident_selected_fastplaid_ratio_delta_vs_baseline")
        for row in comparisons
    )
    ratio_vs_baseline = _floats(
        row.get("gpu_resident_selected_fastplaid_ratio_vs_baseline")
        for row in comparisons
    )
    widened = [
        row
        for row in comparisons
        if _candidate_k(row, "effective_candidate_k")
        > _candidate_k(row, "baseline_effective_candidate_k")
    ]
    return {
        "comparison_count": len(comparisons),
        "widened_comparison_count": len(widened),
        "min_gpu_resident_selected_recall_delta_vs_baseline": (
            min(recall_deltas) if recall_deltas else None
        ),
        "max_gpu_resident_selected_recall_delta_vs_baseline": (
            max(recall_deltas) if recall_deltas else None
        ),
        "mean_gpu_resident_selected_recall_delta_vs_baseline": (
            _mean(recall_deltas)
        ),
        "min_gpu_resident_selected_fastplaid_delta_delta_vs_baseline": (
            min(fastplaid_delta_deltas) if fastplaid_delta_deltas else None
        ),
        "mean_gpu_resident_selected_fastplaid_ratio_delta_vs_baseline": (
            _mean(ratio_deltas)
        ),
        "max_gpu_resident_selected_fastplaid_ratio_vs_baseline": (
            max(ratio_vs_baseline) if ratio_vs_baseline else None
        ),
    }


def _case_source(report: dict[str, Any]) -> str:
    selection = report.get("case_selection", {})
    if not isinstance(selection, dict):
        return ""
    return str(selection.get("source", ""))


def _candidate_window_policy(report: dict[str, Any]) -> str:
    controls = report.get("controls", {})
    if not isinstance(controls, dict):
        return ""
    return str(controls.get("candidate_window_policy", ""))


def _ok_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("rows", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    ]


def _baseline_key(
    report: dict[str, Any],
    row: dict[str, Any],
) -> tuple[object, ...]:
    return (
        _case_source(report),
        row.get("name"),
        row.get("fastplaid_device"),
        row.get("seed"),
        row.get("input_candidate_k"),
    )


def _row_identity(row: dict[str, Any]) -> dict[str, object]:
    return {
        "name": row.get("name"),
        "fastplaid_device": row.get("fastplaid_device"),
        "seed": row.get("seed"),
        "document_count": _shape_int(row, "document_count"),
        "document_vector_count": _shape_int(row, "document_vector_count"),
        "query_count": _shape_int(row, "query_count"),
        "query_vector_count": _shape_int(row, "query_vector_count"),
    }


def _shape_int(row: dict[str, Any], key: str) -> int | None:
    shape = row.get("shape", {})
    if not isinstance(shape, dict):
        return None
    value = shape.get(key)
    return value if isinstance(value, int) else None


def _candidate_k(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    return int(value) if isinstance(value, int) else 0


def _delta(left: object, right: object) -> float | None:
    left_float = _optional_float(left)
    right_float = _optional_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _ratio(numerator: object, denominator: object) -> float | None:
    num = _optional_float(numerator)
    den = _optional_float(denominator)
    if num is None or den is None or den <= 0.0:
        return None
    return num / den


def _optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _floats(values: Iterable[object]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (float, int))
    ]


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _min_optional(values: Iterable[object]) -> float | None:
    floats = _floats(values)
    return min(floats) if floats else None
