"""Metric helpers for resident selected-posting FastPlaid scope rows."""

from __future__ import annotations

from typing import Any


STATUS_OK = "ok"
STATUS_PARTIAL_GPU_UNAVAILABLE = "partial_gpu_unavailable"
STATUS_BLOCKED_FASTPLAID_UNAVAILABLE = "blocked_fastplaid_unavailable"
STATUS_BLOCKED_GPU_RESIDENT_SELECTED_FAILED = (
    "blocked_gpu_resident_selected_posting_exact_rerank_failed"
)


def build_gpu_resident_selected_posting_exact_rerank_vs_fastplaid_comparison(
    *,
    resident_row: dict[str, Any] | None,
    fastplaid_row: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed = _parsed_payload(resident_row)
    fastplaid_batch = _fastplaid_float(fastplaid_row, "query_batch_mean_seconds")
    candidate_seconds = _optional_float(
        parsed.get("resident_candidate_seconds_per_window")
    )
    candidate_cold_seconds = _optional_float(
        parsed.get("resident_candidate_cold_seconds_per_window")
    )
    selected_seconds = _optional_float(
        parsed.get("cpu_selected_centroids_seconds_per_window")
    )
    exact_seconds = _optional_float(
        parsed.get("exact_rerank_topk_seconds_per_window")
    )
    selected_posting_prepare_seconds = _optional_float(
        parsed.get("selected_posting_prepare_extension_call_seconds")
    )
    selected_posting_release_seconds = _optional_float(
        parsed.get("selected_posting_release_extension_call_seconds")
    )
    total_seconds = _optional_float(
        parsed.get(
            "resident_selected_posting_exact_rerank_seconds_per_window"
        )
    )
    cold_total_seconds = _optional_float(
        parsed.get(
            "resident_selected_posting_exact_rerank_cold_seconds_per_window"
        )
    )
    return {
        "status": resident_selected_comparison_status(
            resident_row,
            fastplaid_row,
        ),
        "scope": (
            "gpu_resident_selected_posting_exact_rerank_vs_fastplaid_full_search"
        ),
        "scope_warning": (
            "This is not a public backend comparison: FastPlaid is timed as "
            "full search, while Kayak measures an internal resident "
            "selected-posting candidate primitive plus exact rerank."
        ),
        "candidate_score_count_per_window": _optional_int(
            parsed.get("candidate_score_count_per_window")
        ),
        "candidate_k": _optional_int(parsed.get("candidate_k")),
        "topk_return_count_per_window": _optional_int(
            parsed.get("topk_return_count_per_window")
        ),
        "final_topk_position_agreement": _optional_float(
            parsed.get("final_topk_position_agreement")
        ),
        "candidate_position_agreement_min": _optional_float(
            parsed.get("candidate_position_agreement_min")
        ),
        "exact_score_delta_max_abs": _optional_float(
            parsed.get("exact_score_delta_max_abs")
        ),
        "validation_reference_scores_sent_to_extension": parsed.get(
            "validation_reference_scores_sent_to_extension"
        ),
        "gpu_resident_selected_candidate_seconds_per_window": candidate_seconds,
        "gpu_resident_selected_candidate_cold_seconds_per_window": (
            candidate_cold_seconds
        ),
        "gpu_resident_selected_candidate_generation_kind": parsed.get(
            "candidate_generation_kind"
        ),
        "gpu_resident_selected_candidate_prepare_seconds": (
            selected_posting_prepare_seconds
        ),
        "gpu_resident_selected_candidate_release_seconds": (
            selected_posting_release_seconds
        ),
        "cpu_selected_centroids_seconds_per_window": selected_seconds,
        "gpu_resident_selected_exact_rerank_seconds_per_window": exact_seconds,
        "gpu_resident_selected_exact_rerank_seconds_per_window_total": (
            total_seconds
        ),
        "gpu_resident_selected_exact_rerank_cold_seconds_per_window_total": (
            cold_total_seconds
        ),
        "gpu_resident_selected_exact_rerank_exact_share": _ratio(
            exact_seconds,
            total_seconds,
        ),
        "gpu_resident_selected_cpu_selection_share": _ratio(
            selected_seconds,
            total_seconds,
        ),
        "gpu_resident_selected_candidate_share": _ratio(
            candidate_seconds,
            total_seconds,
        ),
        "fastplaid_query_batch_mean_seconds": fastplaid_batch,
        "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second": (
            _ratio(total_seconds, fastplaid_batch)
        ),
        "gpu_resident_selected_exact_rerank_cold_seconds_per_fastplaid_batch_second": (
            _ratio(cold_total_seconds, fastplaid_batch)
        ),
        "recall_at_k_vs_kayak_exact": _optional_float(
            resident_row.get("recall_at_k_vs_kayak_exact")
            if resident_row
            else None
        ),
    }


def resident_selected_comparison_status(
    resident_row: dict[str, Any] | None,
    fastplaid_row: dict[str, Any] | None,
) -> str:
    if fastplaid_row is None or fastplaid_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_FASTPLAID_UNAVAILABLE
    if resident_row is None:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if resident_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_GPU_RESIDENT_SELECTED_FAILED
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


def _optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator
