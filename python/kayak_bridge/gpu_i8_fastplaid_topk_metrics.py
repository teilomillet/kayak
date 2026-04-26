"""Metric helpers for GPU i8 prepared top-k versus FastPlaid reports."""

from __future__ import annotations

from typing import Any


STATUS_OK = "ok"
STATUS_PARTIAL_GPU_UNAVAILABLE = "partial_gpu_unavailable"
STATUS_BLOCKED_GPU_PREPARED_TOPK_FAILED = "blocked_gpu_prepared_topk_failed"
STATUS_BLOCKED_FASTPLAID_UNAVAILABLE = "blocked_fastplaid_unavailable"


def prepared_handle_topk_derived_metrics(
    *,
    parsed: dict[str, object],
    candidate_generation_per_window: float | None,
    cpu_score_per_window: float | None,
) -> dict[str, float | None]:
    topk_seconds = _optional_float(
        parsed.get("score_extension_call_seconds_per_window")
    )
    candidate_plus_topk = _sum_optional(
        candidate_generation_per_window,
        topk_seconds,
    )
    return {
        "gpu_topk_seconds_per_cpu_same_candidate_score_second": _ratio(
            topk_seconds,
            cpu_score_per_window,
        ),
        "cpu_candidate_generation_plus_gpu_topk_seconds_per_window": (
            candidate_plus_topk
        ),
        "cpu_candidate_generation_plus_gpu_topk_seconds_per_cpu_candidate_generation_plus_cpu_score_second": _ratio(
            candidate_plus_topk,
            _sum_optional(candidate_generation_per_window, cpu_score_per_window),
        ),
    }


def build_gpu_prepared_topk_vs_fastplaid_comparison(
    *,
    prepared_handle_topk_row: dict[str, Any] | None,
    fastplaid_row: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed = _parsed_payload(prepared_handle_topk_row)
    fastplaid_batch = _fastplaid_float(fastplaid_row, "query_batch_mean_seconds")
    fastplaid_query = _fastplaid_float(fastplaid_row, "query_mean_seconds")
    topk_per_window = _optional_float(
        parsed.get("score_extension_call_seconds_per_window")
    )
    candidate_score_count = _optional_float(
        parsed.get("candidate_score_count_per_window")
    )
    topk_return_count = _optional_float(parsed.get("topk_return_count_per_window"))
    cpu_candidate_per_window = _nested_optional_float(
        prepared_handle_topk_row,
        "cpu_i8_multi_window_candidate_generation",
        "mean_seconds_per_window",
    )
    cpu_score_per_window = _nested_optional_float(
        prepared_handle_topk_row,
        "cpu_i8_multi_window_same_candidate_reference",
        "mean_seconds_per_window",
    )
    cpu_candidate_plus_topk = _sum_optional(
        cpu_candidate_per_window,
        topk_per_window,
    )
    cpu_candidate_plus_score = _sum_optional(
        cpu_candidate_per_window,
        cpu_score_per_window,
    )

    return {
        "status": prepared_topk_comparison_status(
            prepared_handle_topk_row,
            fastplaid_row,
        ),
        "scope": "gpu_prepared_handle_topk_rerank_boundary_vs_fastplaid_full_search",
        "scope_warning": (
            "This is still not an apples-to-apples backend comparison: "
            "FastPlaid is timed as full search, while the Kayak GPU row starts "
            "after CPU candidate generation and measures an internal explicit "
            "prepared-handle rerank/top-k return boundary."
        ),
        "candidate_score_count_per_window": (
            int(candidate_score_count)
            if candidate_score_count is not None
            else None
        ),
        "topk_return_count_per_window": (
            int(topk_return_count) if topk_return_count is not None else None
        ),
        "topk_position_agreement": _optional_float(
            parsed.get("topk_position_agreement")
        ),
        "score_delta_max_abs": _optional_float(parsed.get("score_delta_max_abs")),
        "score_delta_tolerance": _optional_float(
            parsed.get("score_delta_tolerance")
        ),
        "gpu_prepared_handle_topk_seconds_per_window": topk_per_window,
        "cpu_candidate_generation_seconds_per_window": cpu_candidate_per_window,
        "cpu_same_candidate_score_seconds_per_window": cpu_score_per_window,
        "cpu_candidate_generation_plus_gpu_topk_seconds_per_window": (
            cpu_candidate_plus_topk
        ),
        "cpu_candidate_generation_plus_cpu_score_seconds_per_window": (
            cpu_candidate_plus_score
        ),
        "fastplaid_query_batch_mean_seconds": fastplaid_batch,
        "fastplaid_query_mean_seconds": fastplaid_query,
        "gpu_prepared_handle_topk_seconds_per_fastplaid_batch_second": _ratio(
            topk_per_window,
            fastplaid_batch,
        ),
        "cpu_candidate_generation_plus_gpu_topk_seconds_per_fastplaid_batch_second": _ratio(
            cpu_candidate_plus_topk,
            fastplaid_batch,
        ),
        "gpu_prepared_handle_topk_seconds_per_cpu_same_candidate_score_second": _ratio(
            topk_per_window,
            cpu_score_per_window,
        ),
        "cpu_candidate_generation_plus_gpu_topk_seconds_per_cpu_candidate_generation_plus_cpu_score_second": _ratio(
            cpu_candidate_plus_topk,
            cpu_candidate_plus_score,
        ),
    }


def prepared_topk_comparison_status(
    prepared_handle_topk_row: dict[str, Any] | None,
    fastplaid_row: dict[str, Any] | None,
) -> str:
    if fastplaid_row is None or fastplaid_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_FASTPLAID_UNAVAILABLE
    if prepared_handle_topk_row is None:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if prepared_handle_topk_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_GPU_PREPARED_TOPK_FAILED
    return STATUS_OK


def _parsed_payload(row: dict[str, object] | None) -> dict[str, object]:
    parsed = None if row is None else row.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def _fastplaid_float(
    fastplaid_row: dict[str, Any] | None,
    key: str,
) -> float | None:
    return _optional_float(
        None if fastplaid_row is None else fastplaid_row.get(key)
    )


def _nested_optional_float(
    row: dict[str, Any] | None,
    parent_key: str,
    child_key: str,
) -> float | None:
    parent = None if row is None else row.get(parent_key)
    if not isinstance(parent, dict):
        return None
    return _optional_float(parent.get(child_key))


def _optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _sum_optional(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator
