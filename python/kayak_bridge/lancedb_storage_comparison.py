"""Owns storage-controlled comparisons on top of a LanceDB table.

The purpose is to separate storage choice from search choice:
- both branches read the same stored vectors from LanceDB
- one branch uses LanceDB multivector search
- the other branch materializes a Kayak packed index from those rows and uses
  Kayak exact search
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import kayak

from .kayak_task_benchmark import benchmark_queries_with_kayak_exact
from .lancedb_arrow_bridge import table_to_packed_kayak_index
from .lancedb_benchmark import (
    _build_table_rows,
    _directory_byte_size,
    _filter_zero_vectors,
    _require_lancedb,
    _search_doc_ids,
    _validate_unit_norm_vectors,
)
from .judged_metrics import summarize_ranked_task
@dataclass(frozen=True, slots=True)
class LanceDbStorageSearchSummary:
    search_engine: str
    engine_version: str
    index_kind: str
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    mean_search_seconds: float


@dataclass(frozen=True, slots=True)
class LanceDbStorageComparisonSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    query_count: int
    document_count: int
    nominal_query_vector_count: int
    nominal_document_vector_count: int
    stored_document_vector_count_total: int
    vector_dim: int
    storage_engine: str
    storage_engine_version: str
    storage_byte_size: int
    kayak_load_from_lancedb_seconds: float
    zero_document_vector_count_filtered: int
    zero_query_vector_count_filtered: int
    vector_unit_norm_max_error: float
    lancedb_scan: LanceDbStorageSearchSummary
    kayak_exact_from_lancedb: LanceDbStorageSearchSummary
    lancedb_scan_latency_ratio_vs_kayak_from_lancedb: float

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_task_with_lancedb_storage_compare(
    *,
    task: Mapping[str, Any],
    database_root: Path,
    table_name: str,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
    unit_norm_tolerance: float = 1e-3,
) -> LanceDbStorageComparisonSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    lancedb, pa = _require_lancedb()
    norm_error = _validate_unit_norm_vectors(task, tolerance=unit_norm_tolerance)
    (
        filtered_task,
        zero_document_vector_count,
        zero_query_vector_count,
        stored_document_vector_count_total,
        _stored_query_vector_count_total,
    ) = _filter_zero_vectors(task)

    if database_root.exists():
        shutil.rmtree(database_root)
    database_root.mkdir(parents=True, exist_ok=True)

    vector_dim = int(filtered_task["vector_dim"])
    db = lancedb.connect(str(database_root))
    schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.list_(pa.float32(), vector_dim))),
        ]
    )
    table = db.create_table(
        table_name,
        data=_build_table_rows(filtered_task),
        schema=schema,
        mode="overwrite",
    )

    query_matrices = tuple(
        filtered_query["vectors"] for filtered_query in filtered_task["queries"]
    )
    for _ in range(warmup_iterations):
        for query_matrix in query_matrices:
            _search_doc_ids(table, query_matrix, int(filtered_task["k"]))

    elapsed_seconds: list[float] = []
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for measurement_iteration in range(measurement_iterations):
        for query_index, query_matrix in enumerate(query_matrices):
            start = time.perf_counter()
            ranked_doc_ids = _search_doc_ids(table, query_matrix, int(filtered_task["k"]))
            elapsed_seconds.append(time.perf_counter() - start)
            if measurement_iteration == 0:
                ranked_doc_ids_by_query.append(ranked_doc_ids)
            elif query_index >= len(ranked_doc_ids_by_query):
                raise AssertionError("ranked query capture drifted during measurement")

    lancedb_metrics = summarize_ranked_task(
        task=filtered_task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    lancedb_scan_summary = LanceDbStorageSearchSummary(
        search_engine="lancedb_scan",
        engine_version=str(lancedb.__version__),
        index_kind="none",
        primary_value=lancedb_metrics.primary_value,
        mean_ndcg_at_k=lancedb_metrics.mean_ndcg_at_k,
        mean_reciprocal_rank=lancedb_metrics.mean_reciprocal_rank,
        mean_recall_at_k=lancedb_metrics.mean_recall_at_k,
        success_rate_at_k=lancedb_metrics.success_rate_at_k,
        mean_search_seconds=float(sum(elapsed_seconds) / len(elapsed_seconds)),
    )

    load_start = time.perf_counter()
    kayak_index = table_to_packed_kayak_index(table)
    kayak_load_seconds = time.perf_counter() - load_start
    kayak_queries = tuple(
        kayak.query(query["vectors"], text=query["text"])
        for query in filtered_task["queries"]
    )
    kayak_summary = benchmark_queries_with_kayak_exact(
        task=filtered_task,
        index=kayak_index,
        queries=kayak_queries,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        backend=kayak.MOJO_EXACT_CPU_BACKEND,
    )
    kayak_storage_summary = LanceDbStorageSearchSummary(
        search_engine="kayak_exact_from_lancedb",
        engine_version=kayak_summary.engine_version,
        index_kind=kayak_summary.index_kind,
        primary_value=kayak_summary.primary_value,
        mean_ndcg_at_k=kayak_summary.mean_ndcg_at_k,
        mean_reciprocal_rank=kayak_summary.mean_reciprocal_rank,
        mean_recall_at_k=kayak_summary.mean_recall_at_k,
        success_rate_at_k=kayak_summary.success_rate_at_k,
        mean_search_seconds=kayak_summary.mean_search_seconds,
    )

    return LanceDbStorageComparisonSummary(
        dataset_id=str(filtered_task["dataset_id"]),
        model_name=str(filtered_task["model_name"]),
        family=str(filtered_task["family"]),
        slice_name=str(filtered_task["slice_name"]),
        primary_metric=str(filtered_task["primary_metric"]),
        k=int(filtered_task["k"]),
        query_count=len(filtered_task["queries"]),
        document_count=len(filtered_task["documents"]),
        nominal_query_vector_count=int(task["nominal_query_vector_count"]),
        nominal_document_vector_count=int(task["nominal_document_vector_count"]),
        stored_document_vector_count_total=stored_document_vector_count_total,
        vector_dim=vector_dim,
        storage_engine="lancedb",
        storage_engine_version=str(lancedb.__version__),
        storage_byte_size=_directory_byte_size(database_root),
        kayak_load_from_lancedb_seconds=kayak_load_seconds,
        zero_document_vector_count_filtered=zero_document_vector_count,
        zero_query_vector_count_filtered=zero_query_vector_count,
        vector_unit_norm_max_error=norm_error,
        lancedb_scan=lancedb_scan_summary,
        kayak_exact_from_lancedb=kayak_storage_summary,
        lancedb_scan_latency_ratio_vs_kayak_from_lancedb=(
            lancedb_scan_summary.mean_search_seconds
            / kayak_storage_summary.mean_search_seconds
        ),
    )
