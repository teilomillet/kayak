"""Owns the LanceDB multivector benchmark runner for encoded task JSON.

This module keeps the external-engine baseline narrow:
- consume the same encoded task JSON already used in-repo
- validate the cosine-vs-dot-product compatibility assumption explicitly
- report judged metrics and storage context in one machine-readable summary
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from .judged_metrics import summarize_ranked_task


def _require_lancedb() -> tuple[Any, Any]:
    try:
        import lancedb
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - exercised in integration runs.
        raise RuntimeError(
            "lancedb benchmark requires optional dependencies; "
            "run via `uv run --python 3.11 --with lancedb ...`"
        ) from exc
    return lancedb, pa


def _require_faiss_for_indexing() -> None:
    try:
        import faiss  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in integration runs.
        raise RuntimeError(
            "indexed lancedb benchmark requires faiss; "
            "run via `uv run --python 3.11 --with lancedb --with faiss-cpu ...`"
        ) from exc


def _directory_byte_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _vectors_unit_norm_error(task: Mapping[str, Any]) -> float:
    max_error = 0.0

    for document in task["documents"]:
        for vector in document["vectors"]:
            norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
            if norm == 0.0:
                continue
            max_error = max(max_error, abs(norm - 1.0))

    for query in task["queries"]:
        for vector in query["vectors"]:
            norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
            if norm == 0.0:
                continue
            max_error = max(max_error, abs(norm - 1.0))

    return max_error


def _validate_unit_norm_vectors(
    task: Mapping[str, Any], *, tolerance: float
) -> float:
    max_error = _vectors_unit_norm_error(task)
    if max_error > tolerance:
        raise ValueError(
            "task vectors are not unit-normalized enough for a cosine-vs-dot "
            f"comparison: max norm error {max_error:.6f} > tolerance {tolerance:.6f}"
        )
    return max_error


def _build_table_rows(task: Mapping[str, Any]) -> list[dict[str, object]]:
    rows = []
    for document in task["documents"]:
        rows.append(
            {
                "doc_id": str(document["doc_id"]),
                "text": str(document["text"]),
                "vector": np.asarray(document["vectors"], dtype=np.float32).tolist(),
            }
        )
    return rows


def _filter_zero_vectors(
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int, int, int]:
    filtered_documents = []
    filtered_queries = []
    zero_document_vector_count = 0
    zero_query_vector_count = 0
    stored_document_vector_count_total = 0
    stored_query_vector_count_total = 0

    for document in task["documents"]:
        kept_vectors = []
        for vector in document["vectors"]:
            if float(np.linalg.norm(np.asarray(vector, dtype=np.float32))) == 0.0:
                zero_document_vector_count += 1
                continue
            kept_vectors.append(vector)

        if not kept_vectors:
            raise ValueError(
                f"document {document['doc_id']} lost all vectors after zero filtering"
            )

        stored_document_vector_count_total += len(kept_vectors)
        filtered_documents.append(
            {
                **document,
                "vector_count": len(kept_vectors),
                "vectors": kept_vectors,
            }
        )

    for query in task["queries"]:
        kept_vectors = []
        for vector in query["vectors"]:
            if float(np.linalg.norm(np.asarray(vector, dtype=np.float32))) == 0.0:
                zero_query_vector_count += 1
                continue
            kept_vectors.append(vector)

        if not kept_vectors:
            raise ValueError(
                f"query {query['query_id']} lost all vectors after zero filtering"
            )

        stored_query_vector_count_total += len(kept_vectors)
        filtered_queries.append(
            {
                **query,
                "vector_count": len(kept_vectors),
                "vectors": kept_vectors,
            }
        )

    filtered_task = dict(task)
    filtered_task["documents"] = filtered_documents
    filtered_task["queries"] = filtered_queries
    filtered_task["nominal_document_vector_count"] = round(
        stored_document_vector_count_total / float(len(filtered_documents))
    )
    filtered_task["nominal_query_vector_count"] = round(
        stored_query_vector_count_total / float(len(filtered_queries))
    )

    return (
        filtered_task,
        zero_document_vector_count,
        zero_query_vector_count,
        stored_document_vector_count_total,
        stored_query_vector_count_total,
    )


def _search_doc_ids(table: Any, query_vectors: np.ndarray, k: int) -> tuple[str, ...]:
    rows = table.search(query_vectors).limit(k).to_arrow().to_pylist()
    return tuple(str(row["doc_id"]) for row in rows)


@dataclass(frozen=True, slots=True)
class LanceDbBenchmarkSummary:
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
    stored_query_vector_count_mean: int
    stored_document_vector_count_mean: int
    stored_document_vector_count_total: int
    vector_dim: int
    engine: str
    engine_version: str
    index_kind: str
    vector_metric: str
    storage_byte_size: int
    bytes_per_document: float
    bytes_per_vector: float
    query_warmup_iterations: int
    query_measurement_iterations: int
    vector_unit_norm_max_error: float
    zero_document_vector_count_filtered: int
    zero_query_vector_count_filtered: int

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_task_with_lancedb(
    *,
    task: Mapping[str, Any],
    database_root: Path,
    table_name: str,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
    build_index: bool = False,
    unit_norm_tolerance: float = 1e-3,
) -> LanceDbBenchmarkSummary:
    lancedb, pa = _require_lancedb()

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    norm_error = _validate_unit_norm_vectors(task, tolerance=unit_norm_tolerance)
    (
        filtered_task,
        zero_document_vector_count,
        zero_query_vector_count,
        stored_document_vector_count_total,
        stored_query_vector_count_total,
    ) = _filter_zero_vectors(task)

    if database_root.exists():
        shutil.rmtree(database_root)
    database_root.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(database_root))
    vector_dim = int(filtered_task["vector_dim"])
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

    index_kind = "none"
    if build_index:
        _require_faiss_for_indexing()
        table.create_index(vector_column_name="vector", metric="cosine")
        index_kind = "ivf_pq"

    query_matrices = tuple(
        np.asarray(query["vectors"], dtype=np.float32)
        for query in filtered_task["queries"]
    )

    for _ in range(warmup_iterations):
        for query_matrix in query_matrices:
            _search_doc_ids(table, query_matrix, int(task["k"]))

    elapsed_seconds: list[float] = []
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for measurement_iteration in range(measurement_iterations):
        for query_index, query_matrix in enumerate(query_matrices):
            start = time.perf_counter()
            ranked_doc_ids = _search_doc_ids(
                table, query_matrix, int(filtered_task["k"])
            )
            elapsed_seconds.append(time.perf_counter() - start)
            if measurement_iteration == 0:
                ranked_doc_ids_by_query.append(ranked_doc_ids)
            elif query_index >= len(ranked_doc_ids_by_query):
                raise AssertionError("ranked query capture drifted during measurement")

    task_metrics = summarize_ranked_task(
        task=filtered_task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    storage_byte_size = _directory_byte_size(database_root)
    document_count = int(task_metrics.document_count)

    return LanceDbBenchmarkSummary(
        dataset_id=str(filtered_task["dataset_id"]),
        model_name=str(filtered_task["model_name"]),
        family=str(filtered_task["family"]),
        slice_name=str(filtered_task["slice_name"]),
        primary_metric=task_metrics.primary_metric,
        primary_value=task_metrics.primary_value,
        mean_ndcg_at_k=task_metrics.mean_ndcg_at_k,
        mean_reciprocal_rank=task_metrics.mean_reciprocal_rank,
        mean_recall_at_k=task_metrics.mean_recall_at_k,
        success_rate_at_k=task_metrics.success_rate_at_k,
        mean_search_seconds=float(sum(elapsed_seconds) / len(elapsed_seconds)),
        k=task_metrics.k,
        query_count=task_metrics.query_count,
        document_count=document_count,
        nominal_query_vector_count=int(task["nominal_query_vector_count"]),
        nominal_document_vector_count=int(task["nominal_document_vector_count"]),
        stored_query_vector_count_mean=round(
            stored_query_vector_count_total / float(task_metrics.query_count)
        ),
        stored_document_vector_count_mean=round(
            stored_document_vector_count_total / float(document_count)
        ),
        stored_document_vector_count_total=stored_document_vector_count_total,
        vector_dim=vector_dim,
        engine="lancedb",
        engine_version=str(lancedb.__version__),
        index_kind=index_kind,
        vector_metric="cosine",
        storage_byte_size=storage_byte_size,
        bytes_per_document=float(storage_byte_size) / float(document_count),
        bytes_per_vector=float(storage_byte_size)
        / float(stored_document_vector_count_total),
        query_warmup_iterations=warmup_iterations,
        query_measurement_iterations=measurement_iterations,
        vector_unit_norm_max_error=norm_error,
        zero_document_vector_count_filtered=zero_document_vector_count,
        zero_query_vector_count_filtered=zero_query_vector_count,
    )
