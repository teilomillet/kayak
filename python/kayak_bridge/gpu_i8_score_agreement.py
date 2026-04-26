"""Owns GPU i8 score agreement tolerance for benchmark probes.

The tolerance is intentionally explicit because CPU and GPU paths both use
Float32 score scalars, but they do not reduce dot products in the same order.
"""

from __future__ import annotations


GPU_I8_SCORE_DELTA_TOLERANCE_FLOOR = 1.0e-4
GPU_I8_SCORE_DELTA_TOLERANCE_PER_QUERY_VECTOR = 1.0e-5


def gpu_i8_score_delta_tolerance(query_vector_count: int) -> float:
    if query_vector_count <= 0:
        raise ValueError("query_vector_count must be positive")
    return max(
        GPU_I8_SCORE_DELTA_TOLERANCE_FLOOR,
        GPU_I8_SCORE_DELTA_TOLERANCE_PER_QUERY_VECTOR
        * float(query_vector_count),
    )


def gpu_i8_score_agreement_ok(
    *,
    score_delta_max_abs: float,
    query_vector_count: int,
) -> bool:
    return score_delta_max_abs <= gpu_i8_score_delta_tolerance(query_vector_count)


def gpu_i8_score_agreement_fields(
    *,
    score_delta_max_abs: float,
    query_vector_count: int,
) -> dict[str, float | bool]:
    tolerance = gpu_i8_score_delta_tolerance(query_vector_count)
    return {
        "score_delta_max_abs": score_delta_max_abs,
        "score_delta_tolerance": tolerance,
        "score_agreement_ok": score_delta_max_abs <= tolerance,
    }
