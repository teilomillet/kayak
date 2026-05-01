"""Owns judged task benchmarks for Tachiom-style retrieval.

The benchmark consumes encoded task JSON with aligned document token IDs. It
intentionally refuses vector-only tasks because token-aware clustering without
token IDs would not be the algorithm described by the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, TOKEN_ID_DTYPE, VECTOR_DTYPE
from .judged_metrics import summarize_ranked_task
from .kayak_task_benchmark import rank_task_with_kayak_exact
from .tachiom_full import TachiomTacHnswResidualPqIndex
from .tachiom_hnsw import TachiomHnswConfig, TachiomTacHnswIndex
from .tachiom_index import TachiomTacIndex
from .tachiom_metrics import mean_candidate_set_recall_at_k, mean_recall_at_k
from .tachiom_pq import TachiomResidualPqConfig, TachiomResidualPqIndex
from .tachiom_types import TachiomTacConfig


class _TaskSearchIndex(Protocol):
    doc_ids: tuple[str, ...]
    document_count: int
    vector_dim: int
    centroid_count: int
    posting_count: int
    index_bytes: int
    index_kind: str

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]: ...

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class TachiomTaskBenchmarkSummary:
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
    exact_primary_value: float
    exact_mean_ndcg_at_k: float
    exact_mean_recall_at_k: float
    candidate_recall_at_k_vs_exact: float
    final_recall_at_k_vs_exact: float
    query_batch_mean_seconds: float
    query_mean_seconds: float
    query_qps: float
    build_seconds: float
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
    index_kind: str
    vector_metric: str
    index_bytes: int
    centroid_count: int
    posting_count: int
    candidate_window_min_count: int
    candidate_window_mean_count: float
    candidate_window_max_count: int
    query_warmup_iterations: int
    query_measurement_iterations: int
    engine_config: dict[str, object]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_task_with_tachiom(
    task: Mapping[str, Any],
    *,
    engine: str,
    tac_config: TachiomTacConfig,
    hnsw_config: TachiomHnswConfig | None = None,
    pq_config: TachiomResidualPqConfig | None = None,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    exact_backend: str = "numpy_reference",
) -> TachiomTaskBenchmarkSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    final_k = int(task["k"])
    query_matrices = _task_query_matrices(task)
    started_at = time.perf_counter()
    index = _build_index(
        task,
        engine=engine,
        tac_config=tac_config,
        hnsw_config=hnsw_config,
        pq_config=pq_config,
        final_k=final_k,
    )
    build_seconds = time.perf_counter() - started_at

    exact_ranked_doc_ids = rank_task_with_kayak_exact(
        task,
        backend=exact_backend,
    )
    exact_metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=exact_ranked_doc_ids,
    )
    reference_positions = _doc_id_rows_to_positions(
        exact_ranked_doc_ids,
        doc_ids=index.doc_ids,
    )

    for _ in range(warmup_iterations):
        _search_task_positions(index, query_matrices, final_k=final_k)

    durations: list[float] = []
    first_final_positions: tuple[tuple[int, ...], ...] | None = None
    first_candidate_positions: tuple[tuple[int, ...], ...] | None = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = _search_task_positions(index, query_matrices, final_k=final_k)
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = _candidate_task_positions(
                index,
                query_matrices,
                final_k=final_k,
            )

    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("Tachiom task benchmark produced no rankings")

    ranked_doc_ids = _positions_to_doc_id_rows(
        first_final_positions,
        doc_ids=index.doc_ids,
    )
    task_metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids,
    )
    stats = _measurement_stats(durations)
    vector_counts = _task_vector_counts(task)
    candidate_counts = [len(row) for row in first_candidate_positions]
    query_count = len(query_matrices)
    batch_mean_seconds = stats["mean_seconds"]

    return TachiomTaskBenchmarkSummary(
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
        exact_primary_value=exact_metrics.primary_value,
        exact_mean_ndcg_at_k=exact_metrics.mean_ndcg_at_k,
        exact_mean_recall_at_k=exact_metrics.mean_recall_at_k,
        candidate_recall_at_k_vs_exact=mean_candidate_set_recall_at_k(
            candidate_positions_by_query=first_candidate_positions,
            reference_positions_by_query=reference_positions,
            k=final_k,
        ),
        final_recall_at_k_vs_exact=mean_recall_at_k(
            candidate_positions_by_query=first_final_positions,
            reference_positions_by_query=reference_positions,
            k=final_k,
        ),
        query_batch_mean_seconds=batch_mean_seconds,
        query_mean_seconds=batch_mean_seconds / float(query_count),
        query_qps=float(query_count) / batch_mean_seconds,
        build_seconds=build_seconds,
        k=task_metrics.k,
        query_count=query_count,
        document_count=task_metrics.document_count,
        nominal_query_vector_count=int(task["nominal_query_vector_count"]),
        nominal_document_vector_count=int(task["nominal_document_vector_count"]),
        query_vector_count_mean=vector_counts["query_vector_count_mean"],
        document_vector_count_mean=vector_counts["document_vector_count_mean"],
        document_vector_count_total=vector_counts["document_vector_count_total"],
        vector_dim=int(task["vector_dim"]),
        engine=engine,
        index_kind=index.index_kind,
        vector_metric="dot_product",
        index_bytes=int(index.index_bytes),
        centroid_count=int(index.centroid_count),
        posting_count=int(index.posting_count),
        candidate_window_min_count=min(candidate_counts),
        candidate_window_mean_count=float(sum(candidate_counts) / len(candidate_counts)),
        candidate_window_max_count=max(candidate_counts),
        query_warmup_iterations=warmup_iterations,
        query_measurement_iterations=measurement_iterations,
        engine_config=_engine_config_json(
            tac_config=tac_config,
            hnsw_config=hnsw_config,
            pq_config=pq_config,
        ),
    )


def _build_index(
    task: Mapping[str, Any],
    *,
    engine: str,
    tac_config: TachiomTacConfig,
    hnsw_config: TachiomHnswConfig | None,
    pq_config: TachiomResidualPqConfig | None,
    final_k: int,
) -> _TaskSearchIndex:
    doc_ids, doc_offsets, token_vectors, token_ids = _packed_task_documents(task)
    tac_index = TachiomTacIndex.build_packed(
        doc_ids=doc_ids,
        doc_offsets=doc_offsets,
        token_vectors=token_vectors,
        token_ids=token_ids,
        config=tac_config,
        final_k=final_k,
    )
    if engine == "tachiom_tac":
        return tac_index
    if engine == "tachiom_tac_hnsw":
        return TachiomTacHnswIndex.from_tac_index(
            tac_index,
            config=_require_hnsw_config(hnsw_config, engine),
        )
    if engine == "tachiom_tac_pq":
        return TachiomResidualPqIndex.from_tac_index(
            tac_index,
            config=_require_pq_config(pq_config, engine),
        )
    if engine == "tachiom_tac_hnsw_pq":
        return TachiomTacHnswResidualPqIndex.from_tac_index(
            tac_index,
            hnsw_config=_require_hnsw_config(hnsw_config, engine),
            pq_config=_require_pq_config(pq_config, engine),
        )
    raise ValueError(f"unsupported Tachiom task engine: {engine}")


def _packed_task_documents(
    task: Mapping[str, Any],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    documents = task["documents"]
    vector_dim = int(task["vector_dim"])
    doc_ids: list[str] = []
    offsets = [0]
    vector_rows: list[np.ndarray] = []
    token_id_rows: list[np.ndarray] = []
    for document in documents:
        if "token_ids" not in document:
            raise ValueError(
                "Tachiom task benchmark requires document token_ids aligned "
                "with document vectors"
            )
        vectors = np.asarray(document["vectors"], dtype=VECTOR_DTYPE)
        if vectors.ndim != 2 or int(vectors.shape[1]) != vector_dim:
            raise ValueError("document vectors must have shape tokens x vector_dim")
        token_ids = np.asarray(document["token_ids"], dtype=TOKEN_ID_DTYPE).reshape(-1)
        if int(token_ids.shape[0]) != int(vectors.shape[0]):
            raise ValueError("document token_ids must align with document vectors")
        doc_ids.append(str(document["doc_id"]))
        vector_rows.append(vectors)
        token_id_rows.append(token_ids)
        offsets.append(offsets[-1] + int(vectors.shape[0]))
    if not vector_rows:
        raise ValueError("Tachiom task benchmark requires at least one document")
    return (
        tuple(doc_ids),
        np.asarray(offsets, dtype=INDEX_OFFSET_DTYPE),
        np.ascontiguousarray(np.concatenate(vector_rows, axis=0), dtype=VECTOR_DTYPE),
        np.ascontiguousarray(np.concatenate(token_id_rows, axis=0), dtype=TOKEN_ID_DTYPE),
    )


def _task_query_matrices(task: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    vector_dim = int(task["vector_dim"])
    matrices: list[np.ndarray] = []
    for query in task["queries"]:
        matrix = np.asarray(query["vectors"], dtype=VECTOR_DTYPE)
        if matrix.ndim != 2 or int(matrix.shape[1]) != vector_dim:
            raise ValueError("query vectors must have shape tokens x vector_dim")
        matrices.append(np.ascontiguousarray(matrix, dtype=VECTOR_DTYPE))
    if not matrices:
        raise ValueError("Tachiom task benchmark requires at least one query")
    return tuple(matrices)


def _search_task_positions(
    index: _TaskSearchIndex,
    query_matrices: Sequence[np.ndarray],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        index.search_batch_positions(query[None, :, :], final_k=final_k)[0]
        for query in query_matrices
    )


def _candidate_task_positions(
    index: _TaskSearchIndex,
    query_matrices: Sequence[np.ndarray],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        index.candidate_positions_batch(query[None, :, :], final_k=final_k)[0]
        for query in query_matrices
    )


def _positions_to_doc_id_rows(
    positions_by_query: Sequence[Sequence[int]],
    *,
    doc_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(str(doc_ids[int(position)]) for position in row)
        for row in positions_by_query
    )


def _doc_id_rows_to_positions(
    ranked_doc_ids_by_query: Sequence[Sequence[str]],
    *,
    doc_ids: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    positions_by_doc_id = {doc_id: position for position, doc_id in enumerate(doc_ids)}
    return tuple(
        tuple(positions_by_doc_id[str(doc_id)] for doc_id in row)
        for row in ranked_doc_ids_by_query
    )


def _task_vector_counts(task: Mapping[str, Any]) -> dict[str, int]:
    document_total = sum(int(document["vector_count"]) for document in task["documents"])
    query_total = sum(int(query["vector_count"]) for query in task["queries"])
    return {
        "document_vector_count_total": document_total,
        "document_vector_count_mean": round(
            document_total / float(len(task["documents"]))
        ),
        "query_vector_count_mean": round(query_total / float(len(task["queries"]))),
    }


def _measurement_stats(durations: Sequence[float]) -> dict[str, float]:
    if not durations:
        raise ValueError("durations must not be empty")
    ordered = sorted(float(duration) for duration in durations)
    return {
        "mean_seconds": float(sum(ordered) / len(ordered)),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def _require_hnsw_config(
    config: TachiomHnswConfig | None,
    engine: str,
) -> TachiomHnswConfig:
    if config is None:
        raise ValueError(f"{engine} requires hnsw_config")
    return config


def _require_pq_config(
    config: TachiomResidualPqConfig | None,
    engine: str,
) -> TachiomResidualPqConfig:
    if config is None:
        raise ValueError(f"{engine} requires pq_config")
    return config


def _engine_config_json(
    *,
    tac_config: TachiomTacConfig,
    hnsw_config: TachiomHnswConfig | None,
    pq_config: TachiomResidualPqConfig | None,
) -> dict[str, object]:
    config: dict[str, object] = {"tac": asdict(tac_config)}
    if hnsw_config is not None:
        config["hnsw"] = asdict(hnsw_config)
    if pq_config is not None:
        config["pq"] = asdict(pq_config)
    return config
