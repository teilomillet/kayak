"""Owns Python-driven Kayak exact benchmarks for encoded task JSON.

This module exists for apples-to-apples external comparisons where both
systems should consume the same task JSON under the same Python driver. It
does not replace the standalone Mojo benchmark entrypoints, which remain the
repo's lowest-overhead exact measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping, Sequence

import kayak

from .judged_metrics import summarize_ranked_task


def _document_vector_count_total(task: Mapping[str, Any]) -> int:
    return sum(int(document["vector_count"]) for document in task["documents"])


def _query_vector_count_total(task: Mapping[str, Any]) -> int:
    return sum(int(query["vector_count"]) for query in task["queries"])


def _build_index(task: Mapping[str, Any]) -> kayak.LateIndex:
    return kayak.documents(
        [document["doc_id"] for document in task["documents"]],
        [document["vectors"] for document in task["documents"]],
        texts=[document["text"] for document in task["documents"]],
    ).pack()


def _build_queries(task: Mapping[str, Any]) -> tuple[kayak.LateQuery, ...]:
    return tuple(
        kayak.query(query["vectors"], text=query["text"])
        for query in task["queries"]
    )


def rank_task_with_kayak_exact(
    task: Mapping[str, Any],
    *,
    backend: str = kayak.MOJO_EXACT_CPU_BACKEND,
) -> tuple[tuple[str, ...], ...]:
    index = _build_index(task)
    queries = _build_queries(task)
    hits_by_query = kayak.search_batch(
        kayak.query_batch(queries),
        index,
        k=int(task["k"]),
        backend=backend,
    )
    return tuple(
        tuple(hit.doc_id for hit in hits)
        for hits in hits_by_query
    )


def benchmark_queries_with_kayak_exact(
    *,
    task: Mapping[str, Any],
    index: kayak.LateIndex,
    queries: Sequence[kayak.LateQuery],
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
    backend: str = kayak.MOJO_EXACT_CPU_BACKEND,
) -> KayakExactTaskBenchmarkSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    query_batch = kayak.query_batch(queries)
    k = int(task["k"])

    for _ in range(warmup_iterations):
        kayak.search_batch(query_batch, index, k=k, backend=backend)

    elapsed_seconds: list[float] = []
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for measurement_iteration in range(measurement_iterations):
        start = time.perf_counter()
        hits_by_query = kayak.search_batch(query_batch, index, k=k, backend=backend)
        elapsed_seconds.append(
            (time.perf_counter() - start) / float(query_batch.batch_size)
        )

        if measurement_iteration == 0:
            ranked_doc_ids_by_query = [
                tuple(hit.doc_id for hit in hits)
                for hits in hits_by_query
            ]

    task_metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    query_vector_count_total = _query_vector_count_total(task)
    document_vector_count_total = _document_vector_count_total(task)

    return KayakExactTaskBenchmarkSummary(
        dataset_id=str(task["dataset_id"]),
        model_name=str(task["model_name"]),
        family=str(task["family"]),
        slice_name=str(task["slice_name"]),
        primary_metric=task_metrics.primary_metric,
        primary_value=task_metrics.primary_value,
        mean_ndcg_at_k=task_metrics.mean_ndcg_at_k,
        mean_reciprocal_rank=task_metrics.mean_reciprocal_rank,
        mean_recall_at_k=task_metrics.mean_recall_at_k,
        success_rate_at_k=task_metrics.success_rate_at_k,
        mean_search_seconds=float(sum(elapsed_seconds) / len(elapsed_seconds)),
        k=task_metrics.k,
        query_count=task_metrics.query_count,
        document_count=task_metrics.document_count,
        nominal_query_vector_count=int(task["nominal_query_vector_count"]),
        nominal_document_vector_count=int(task["nominal_document_vector_count"]),
        query_vector_count_mean=round(
            query_vector_count_total / float(task_metrics.query_count)
        ),
        document_vector_count_mean=round(
            document_vector_count_total / float(task_metrics.document_count)
        ),
        document_vector_count_total=document_vector_count_total,
        vector_dim=int(task["vector_dim"]),
        engine="kayak",
        engine_version="python_sdk",
        index_kind="exact_packed",
        vector_metric="dot_product",
        backend=backend,
        query_warmup_iterations=warmup_iterations,
        query_measurement_iterations=measurement_iterations,
    )


@dataclass(frozen=True, slots=True)
class KayakExactTaskBenchmarkSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    mean_search_seconds: float
    k: int
    query_count: int
    document_count: int
    nominal_query_vector_count: int
    nominal_document_vector_count: int
    query_vector_count_mean: int
    document_vector_count_mean: int
    document_vector_count_total: int
    vector_dim: int
    engine: str
    engine_version: str
    index_kind: str
    vector_metric: str
    backend: str
    query_warmup_iterations: int
    query_measurement_iterations: int

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_task_with_kayak_exact(
    task: Mapping[str, Any],
    *,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
    backend: str = kayak.MOJO_EXACT_CPU_BACKEND,
) -> KayakExactTaskBenchmarkSummary:
    index = _build_index(task)
    queries = _build_queries(task)
    return benchmark_queries_with_kayak_exact(
        task=task,
        index=index,
        queries=queries,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        backend=backend,
    )
