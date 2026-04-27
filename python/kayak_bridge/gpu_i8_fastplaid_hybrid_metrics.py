"""Metric helpers for hybrid GPU i8 FastPlaid scope rows."""

from __future__ import annotations

from typing import Any


STATUS_OK = "ok"
STATUS_PARTIAL_GPU_UNAVAILABLE = "partial_gpu_unavailable"
STATUS_BLOCKED_FASTPLAID_UNAVAILABLE = "blocked_fastplaid_unavailable"
STATUS_BLOCKED_GPU_HYBRID_FAILED = "blocked_gpu_hybrid_shortlist_rerank_failed"


def build_gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_comparison(
    *,
    hybrid_row: dict[str, Any] | None,
    fastplaid_row: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed = _parsed_payload(hybrid_row)
    fastplaid_batch = _fastplaid_float(fastplaid_row, "query_batch_mean_seconds")
    hybrid_seconds = _optional_float(
        parsed.get("hybrid_extension_seconds_per_window")
    )
    fused_seconds = _optional_float(
        parsed.get("fused_device_topk_seconds_per_window")
    )
    exact_seconds = _optional_float(
        parsed.get("exact_rerank_topk_seconds_per_window")
    )
    return {
        "status": hybrid_comparison_status(hybrid_row, fastplaid_row),
        "scope": "gpu_fused_shortlist_exact_rerank_vs_fastplaid_full_search",
        "scope_warning": (
            "This is still not a public backend comparison: FastPlaid is timed "
            "as full search, while Kayak measures an internal GPU fused "
            "shortlist plus exact rerank primitive."
        ),
        "candidate_score_count_per_window": _optional_int(
            parsed.get("candidate_score_count_per_window")
        ),
        "shortlist_k": _optional_int(parsed.get("shortlist_k")),
        "topk_return_count_per_window": _optional_int(
            parsed.get("topk_return_count_per_window")
        ),
        "final_topk_position_agreement": _optional_float(
            parsed.get("final_topk_position_agreement")
        ),
        "exact_score_delta_max_abs": _optional_float(
            parsed.get("exact_score_delta_max_abs")
        ),
        "validation_reference_scores_sent_to_extension": parsed.get(
            "validation_reference_scores_sent_to_extension"
        ),
        "gpu_hybrid_fused_shortlist_seconds_per_window": fused_seconds,
        "gpu_hybrid_exact_rerank_seconds_per_window": exact_seconds,
        "gpu_hybrid_seconds_per_window": hybrid_seconds,
        "gpu_hybrid_exact_rerank_share": _ratio(exact_seconds, hybrid_seconds),
        "fastplaid_query_batch_mean_seconds": fastplaid_batch,
        "gpu_hybrid_seconds_per_fastplaid_batch_second": _ratio(
            hybrid_seconds,
            fastplaid_batch,
        ),
        "recall_at_k_vs_kayak_exact": _optional_float(
            hybrid_row.get("recall_at_k_vs_kayak_exact") if hybrid_row else None
        ),
    }


def hybrid_comparison_status(
    hybrid_row: dict[str, Any] | None,
    fastplaid_row: dict[str, Any] | None,
) -> str:
    if fastplaid_row is None or fastplaid_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_FASTPLAID_UNAVAILABLE
    if hybrid_row is None:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if hybrid_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_GPU_HYBRID_FAILED
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
