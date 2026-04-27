"""Profiles PLAID i8 candidate generation on encoded retrieval tasks.

This module owns the real-task profiling boundary for Kayak's internal
PLAID-style i8 path. It does not own dataset downloading, text encoding, or
external baseline execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .dtypes import VECTOR_DTYPE
from .late_index import LateIndex
from .late_ops import MOJO_EXACT_CPU_BACKEND, documents, query, search
from .late_query import LateQuery
from .mojo_exact_cpu import load_module
from .plaid_approx import (
    KayakPlaidApproxConfig,
    KayakPlaidApproxIndex,
    VECTOR_DIM,
)


I8_CODE_BYTES_PER_TOKEN = VECTOR_DIM
I8_SCALE_BYTES_PER_TOKEN = np.dtype(VECTOR_DTYPE).itemsize
NEXTPLAID_4BIT_RESIDUAL_BYTES_PER_TOKEN = 64


@dataclass(frozen=True, slots=True)
class PlaidTaskCandidateProfileControls:
    """Knobs for one task-level candidate-generation profile."""

    centroid_count: int = 128
    centroids_per_query_vector: int = 32
    candidate_k: int = 256
    query_limit: int | None = None
    measurement_iterations: int = 3
    exact_reference: bool = True
    exact_backend: str = MOJO_EXACT_CPU_BACKEND

    def validate(self, *, final_k: int) -> None:
        if self.centroid_count <= 0:
            raise ValueError("centroid_count must be positive")
        if self.centroids_per_query_vector <= 0:
            raise ValueError("centroids_per_query_vector must be positive")
        if self.candidate_k < final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")
        if self.query_limit is not None and self.query_limit <= 0:
            raise ValueError("query_limit must be positive when provided")
        if self.measurement_iterations <= 0:
            raise ValueError("measurement_iterations must be positive")

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VectorCountStats:
    """Min/mean/max vector counts for one side of a retrieval task."""

    min_count: int
    mean_count: float
    max_count: int
    total_count: int

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def vector_count_stats(counts: Sequence[int]) -> VectorCountStats:
    if not counts:
        return VectorCountStats(0, 0.0, 0, 0)
    normalized = [int(count) for count in counts]
    return VectorCountStats(
        min_count=min(normalized),
        mean_count=float(mean(normalized)),
        max_count=max(normalized),
        total_count=sum(normalized),
    )


def candidate_recall_at_k(
    candidate_doc_ids: Sequence[str],
    reference_doc_ids: Sequence[str],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    reference = set(reference_doc_ids[:k])
    if not reference:
        return 0.0
    candidates = set(candidate_doc_ids)
    return len(reference & candidates) / float(len(reference))


def summarize_numeric_fields(
    rows: Sequence[Mapping[str, Any]],
    field_names: Sequence[str],
) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for field_name in field_names:
        values = [
            float(row[field_name])
            for row in rows
            if row.get(field_name) is not None
        ]
        summary[f"{field_name}_mean"] = (
            None if not values else float(sum(values) / len(values))
        )
        summary[f"{field_name}_max"] = None if not values else max(values)
    return summary


def build_late_index_from_task(task: Mapping[str, Any]) -> LateIndex:
    return documents(
        [str(document["doc_id"]) for document in task["documents"]],
        [document["vectors"] for document in task["documents"]],
        texts=[str(document.get("text", "")) for document in task["documents"]],
    ).pack()


def build_late_queries_from_task(task: Mapping[str, Any]) -> tuple[LateQuery, ...]:
    return tuple(
        query(item["vectors"], text=str(item.get("text", "")))
        for item in task["queries"]
    )


def profile_task_plaid_i8_candidate_generation(
    task: Mapping[str, Any],
    controls: PlaidTaskCandidateProfileControls,
) -> dict[str, Any]:
    final_k = int(task["k"])
    controls.validate(final_k=final_k)

    started_at = time.perf_counter()
    late_index = build_late_index_from_task(task)
    late_queries = build_late_queries_from_task(task)
    selected_queries = _limited_queries(late_queries, controls.query_limit)
    index_build_seconds = time.perf_counter() - started_at

    if late_index.vector_dim != VECTOR_DIM:
        raise ValueError("PLAID i8 task profile currently requires vector_dim=128")
    if not selected_queries:
        raise ValueError("task must provide at least one selected query")

    plaid_started_at = time.perf_counter()
    plaid_index = KayakPlaidApproxIndex.from_late_index(
        late_index,
        config=KayakPlaidApproxConfig(
            centroid_count=controls.centroid_count,
            centroids_per_query_vector=controls.centroids_per_query_vector,
            candidate_k=controls.candidate_k,
            payload="i8",
        ),
        final_k=final_k,
    )
    plaid_prepare_seconds = time.perf_counter() - plaid_started_at

    profile_rows = [
        profile_query(
            late_index=late_index,
            plaid_index=plaid_index,
            late_query=late_query,
            query_index=query_index,
            final_k=final_k,
            controls=controls,
        )
        for query_index, late_query in enumerate(selected_queries)
    ]

    document_counts = late_index.vector_counts
    query_counts = [late_query.vector_count for late_query in selected_queries]
    payload = plaid_index.i8_payload_snapshot()

    return {
        "schema_version": 1,
        "benchmark": "plaid_i8_task_candidate_generation_profile",
        "status": "ok",
        "task": task_metadata(task, selected_query_count=len(selected_queries)),
        "controls": controls.to_json_ready(),
        "index": {
            "document_count": late_index.document_count,
            "document_vector_counts": vector_count_stats(
                document_counts
            ).to_json_ready(),
            "vector_dim": late_index.vector_dim,
            "index_build_seconds": index_build_seconds,
            "plaid_i8_prepare_seconds": plaid_prepare_seconds,
            "plaid_i8_index_bytes": plaid_index.index_bytes,
            "candidate_generation_payload_bytes": (
                payload.candidate_generation_byte_counts()
            ),
            "candidate_generation_payload_total_bytes": int(
                sum(payload.candidate_generation_byte_counts().values())
            ),
            "posting_count": payload.posting_count,
            "centroid_count": payload.centroid_count,
        },
        "queries": {
            "query_count": len(selected_queries),
            "query_vector_counts": vector_count_stats(query_counts).to_json_ready(),
        },
        "rows": profile_rows,
        "aggregate": aggregate_profile_rows(profile_rows),
        "measurement_note": (
            "This profiles Kayak's internal PLAID i8 candidate-generation "
            "substeps on an encoded task JSON. Exact reference recall is "
            "optional and intentionally explicit because full-corpus exact "
            "reference can dominate large-corpus runs."
        ),
    }


def profile_query(
    *,
    late_index: LateIndex,
    plaid_index: KayakPlaidApproxIndex,
    late_query: LateQuery,
    query_index: int,
    final_k: int,
    controls: PlaidTaskCandidateProfileControls,
) -> dict[str, Any]:
    query_matrix = np.expand_dims(
        late_query.as_vector_matrix().astype(VECTOR_DTYPE, copy=False),
        axis=0,
    )
    profile_function = (
        load_module().plaid_i8_candidate_generation_profile_prepared_batch_address
    )
    raw_profile = profile_function(
        [
            int(query_matrix.ctypes.data),
            1,
            int(query_matrix.shape[1]),
            controls.centroids_per_query_vector,
            controls.candidate_k,
            controls.measurement_iterations,
            plaid_index._prepared_index,
        ]
    )
    profile = profile_pairs_to_dict(raw_profile[0])
    candidate_positions = plaid_index.i8_candidate_positions_batch(query_matrix)[0]
    candidate_doc_ids = tuple(
        plaid_index.doc_ids[position] for position in candidate_positions
    )
    candidate_vector_count = candidate_document_vector_count(
        plaid_index.document_vector_counts,
        candidate_positions,
    )
    row: dict[str, Any] = {
        "query_index": query_index,
        "query_vector_count": late_query.vector_count,
        "candidate_count": len(candidate_positions),
        "candidate_document_vector_count": candidate_vector_count,
        "candidate_i8_token_payload_bytes_estimate": (
            candidate_vector_count
            * (I8_CODE_BYTES_PER_TOKEN + I8_SCALE_BYTES_PER_TOKEN)
        ),
        "nextplaid_4bit_residual_bytes_estimate": (
            candidate_vector_count * NEXTPLAID_4BIT_RESIDUAL_BYTES_PER_TOKEN
        ),
        "full_exact_document_vector_count": late_index.total_vector_count,
        "profile": profile,
    }
    if controls.exact_reference:
        exact_hits = search(
            late_query,
            late_index,
            k=final_k,
            backend=controls.exact_backend,
        )
        row["candidate_recall_at_k_vs_exact"] = candidate_recall_at_k(
            candidate_doc_ids,
            [hit.doc_id for hit in exact_hits],
            k=final_k,
        )
    return row


def profile_pairs_to_dict(row: Sequence[Sequence[Any]]) -> dict[str, Any]:
    return {str(key): value for key, value in row}


def candidate_document_vector_count(
    document_vector_counts: Sequence[int],
    candidate_positions: Sequence[int],
) -> int:
    total = 0
    for position in candidate_positions:
        total += int(document_vector_counts[int(position)])
    return total


def aggregate_profile_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    profile_rows = [row["profile"] for row in rows]
    summary = summarize_numeric_fields(
        profile_rows,
        (
            "full_candidate_mean_seconds",
            "workspace_full_candidate_mean_seconds",
            "unordered_candidate_mean_seconds",
            "centroid_scoring_mean_seconds",
            "centroid_selection_mean_seconds",
            "posting_accumulation_mean_seconds",
            "final_topk_mean_seconds",
            "unordered_final_topk_mean_seconds",
            "posting_visit_count",
            "touched_document_count",
        ),
    )
    summary.update(
        summarize_numeric_fields(
            rows,
            (
                "candidate_document_vector_count",
                "candidate_i8_token_payload_bytes_estimate",
                "nextplaid_4bit_residual_bytes_estimate",
                "candidate_recall_at_k_vs_exact",
            ),
        )
    )
    summary["query_count"] = len(rows)
    return summary


def task_metadata(
    task: Mapping[str, Any],
    *,
    selected_query_count: int,
) -> dict[str, object]:
    return {
        "dataset_id": str(task.get("dataset_id", "")),
        "model_name": str(task.get("model_name", "")),
        "family": str(task.get("family", "")),
        "slice_name": str(task.get("slice_name", "")),
        "primary_metric": str(task.get("primary_metric", "")),
        "k": int(task["k"]),
        "document_count": len(task["documents"]),
        "query_count": len(task["queries"]),
        "selected_query_count": selected_query_count,
        "nominal_query_vector_count": int(task.get("nominal_query_vector_count", 0)),
        "nominal_document_vector_count": int(
            task.get("nominal_document_vector_count", 0)
        ),
        "vector_dim": int(task.get("vector_dim", 0)),
    }


def _limited_queries(
    queries: Sequence[LateQuery],
    query_limit: int | None,
) -> tuple[LateQuery, ...]:
    if query_limit is None:
        return tuple(queries)
    return tuple(queries[:query_limit])
