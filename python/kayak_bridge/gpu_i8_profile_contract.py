from __future__ import annotations

from typing import Any


PROFILE_STATUS_NOT_MEASURED = "not_measured"
PROFILE_STATUS_EXPLORATORY = "exploratory"
PROFILE_STATUS_DECISION_READY = "decision_ready"
PROFILE_STATUS_FALSIFIED = "falsified"

GPU_TIMING_FIELDS = (
    "build_or_prepare_seconds",
    "host_to_device_seconds",
    "kernel_seconds",
    "device_to_host_seconds",
    "end_to_end_seconds",
)

AGREEMENT_FIELDS = (
    "score_delta_max_abs",
    "candidate_order_agreement",
    "recall_at_k_vs_cpu_i8",
    "recall_at_k_vs_kayak_exact",
)

GPU_PROFILE_HYPOTHESES = (
    {
        "id": "copy_cost_bound",
        "claim": (
            "candidate-window rerank remains useful only if host-device copy "
            "time does not dominate end-to-end time"
        ),
        "falsifies_if": "host_to_device + device_to_host >= end_to_end",
    },
    {
        "id": "kernel_faster_than_cpu_i8",
        "claim": "GPU kernel time beats CPU i8 rerank time on the same candidates",
        "falsifies_if": "kernel_seconds >= cpu_i8_candidate_rerank_seconds",
    },
    {
        "id": "score_agreement",
        "claim": "GPU i8 scores agree with CPU i8 for the same candidate ids",
        "falsifies_if": "score_delta_max_abs exceeds tolerance or order drifts",
    },
)


def gpu_profile_contract(shape: Any, *, candidate_k: int) -> dict[str, object]:
    return {
        "status": PROFILE_STATUS_NOT_MEASURED,
        "hypotheses": list(GPU_PROFILE_HYPOTHESES),
        "timing_fields": list(GPU_TIMING_FIELDS),
        "agreement_fields": list(AGREEMENT_FIELDS),
        "minimum_decision_runs": 3,
        "required_controls": [
            "same query values",
            "same token codes and token scales",
            "same doc offsets",
            "same candidate positions",
            "same vector counts",
            "quiet benchmark wrapper for decision-quality timing",
        ],
        "shape": {
            "query_count": shape.query_count,
            "query_vector_count": shape.query_vector_count,
            "document_count": shape.document_count,
            "document_vector_count": shape.document_vector_count,
            "total_document_vector_count": (
                shape.document_count * shape.document_vector_count
            ),
            "candidate_k": candidate_k,
            "vector_dim": shape.vector_dim,
        },
    }


def gpu_profile_row_template(shape: Any, *, candidate_k: int) -> dict[str, object]:
    row: dict[str, object] = {
        "status": PROFILE_STATUS_NOT_MEASURED,
        "decision_status": PROFILE_STATUS_NOT_MEASURED,
        "run_count": 0,
        "candidate_k": candidate_k,
        "query_count": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "vector_dim": shape.vector_dim,
    }
    for field in GPU_TIMING_FIELDS:
        row[field] = None
    for field in AGREEMENT_FIELDS:
        row[field] = None
    return row


def gpu_profile_decision_status(
    row: dict[str, object],
    *,
    minimum_runs: int = 3,
    max_score_delta: float = 1e-3,
) -> str:
    score_delta = _optional_float(row.get("score_delta_max_abs"))
    if score_delta is not None and score_delta > max_score_delta:
        return PROFILE_STATUS_FALSIFIED
    if row.get("recall_at_k_vs_cpu_i8") not in (None, 1.0):
        return PROFILE_STATUS_FALSIFIED
    if not _has_all_numbers(row, GPU_TIMING_FIELDS):
        return PROFILE_STATUS_EXPLORATORY
    if not _has_all_numbers(row, AGREEMENT_FIELDS):
        return PROFILE_STATUS_EXPLORATORY
    if _optional_int(row.get("run_count")) < minimum_runs:
        return PROFILE_STATUS_EXPLORATORY
    return PROFILE_STATUS_DECISION_READY


def _has_all_numbers(row: dict[str, object], fields: tuple[str, ...]) -> bool:
    return all(_optional_float(row.get(field)) is not None for field in fields)


def _optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _optional_int(value: object) -> int:
    if isinstance(value, int):
        return value
    return 0
