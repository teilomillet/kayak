"""Benchmarks Tachiom indexes built directly from encoded binary snapshots.

This module owns the in-memory bridge from snapshot shards to the current
Tachiom reference implementation. It does not claim paper-scale construction;
full 598M-vector indexing still needs a streaming builder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dtypes import VECTOR_DTYPE
from .encoded_snapshot_loader import (
    PackedSnapshotDocuments,
    SnapshotQueries,
    load_packed_snapshot_documents,
    load_snapshot_manifest,
    load_snapshot_queries,
    snapshot_task_metadata,
)
from .judged_metrics import summarize_ranked_task
from .tachiom_full import TachiomTacHnswResidualPqIndex
from .tachiom_hnsw import TachiomHnswConfig, TachiomTacHnswIndex
from .tachiom_index import TachiomTacIndex
from .tachiom_metrics import mean_candidate_set_recall_at_k, mean_recall_at_k
from .tachiom_pq import TachiomResidualPqConfig, TachiomResidualPqIndex
from .tachiom_types import TachiomTacConfig


@dataclass(frozen=True, slots=True)
class SnapshotTachiomBenchmarkSummary:
    dataset_id: str
    model_name: str
    primary_metric: str
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    exact_primary_value: float | None
    exact_mean_ndcg_at_k: float | None
    exact_mean_recall_at_k: float | None
    candidate_recall_at_k_vs_exact: float | None
    final_recall_at_k_vs_exact: float | None
    query_batch_mean_seconds: float
    query_mean_seconds: float
    query_qps: float
    build_seconds: float
    k: int
    query_count: int
    document_count: int
    document_vector_count_total: int
    document_vector_count_mean: int
    query_vector_count_total: int
    query_vector_count_mean: int
    vector_dim: int
    engine: str
    index_kind: str
    vector_metric: str
    source_vector_dtype: str
    source_token_id_dtype: str
    loaded_payload_bytes: int
    index_bytes: int
    centroid_count: int
    posting_count: int
    candidate_window_min_count: int
    candidate_window_mean_count: float
    candidate_window_max_count: int
    query_warmup_iterations: int
    query_measurement_iterations: int
    engine_config: dict[str, object]
    epistemic_status: dict[str, object]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_snapshot_with_tachiom(
    snapshot_root: Path,
    *,
    engine: str,
    tac_config: TachiomTacConfig,
    hnsw_config: TachiomHnswConfig | None = None,
    pq_config: TachiomResidualPqConfig | None = None,
    document_limit: int | None = None,
    query_limit: int | None = None,
    max_vector_count: int | None = None,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    run_exact: bool = True,
) -> SnapshotTachiomBenchmarkSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    manifest = load_snapshot_manifest(snapshot_root)
    documents = load_packed_snapshot_documents(
        snapshot_root,
        document_limit=document_limit,
        max_vector_count=max_vector_count,
    )
    queries = load_snapshot_queries(snapshot_root, query_limit=query_limit)
    final_k = queries.k

    started_at = time.perf_counter()
    index = _build_index(
        documents,
        engine=engine,
        tac_config=tac_config,
        hnsw_config=hnsw_config,
        pq_config=pq_config,
        final_k=final_k,
    )
    build_seconds = time.perf_counter() - started_at

    query_matrices = queries.query_matrices
    for _ in range(warmup_iterations):
        _search_snapshot_positions(index, query_matrices, final_k=final_k)

    durations: list[float] = []
    first_final_positions: tuple[tuple[int, ...], ...] | None = None
    first_candidate_positions: tuple[tuple[int, ...], ...] | None = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = _search_snapshot_positions(
            index,
            query_matrices,
            final_k=final_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = _candidate_snapshot_positions(
                index,
                query_matrices,
                final_k=final_k,
            )
    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("snapshot benchmark produced no rankings")

    task = snapshot_task_metadata(
        manifest=manifest,
        documents=documents,
        queries=queries,
    )
    ranked_doc_ids = _positions_to_doc_id_rows(
        first_final_positions,
        doc_ids=documents.doc_ids,
    )
    task_metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids,
    )

    exact_metrics = None
    reference_positions = None
    if run_exact:
        reference_positions = _exact_search_snapshot_positions(
            documents=documents,
            queries=query_matrices,
            final_k=final_k,
        )
        exact_doc_ids = _positions_to_doc_id_rows(
            reference_positions,
            doc_ids=documents.doc_ids,
        )
        exact_metrics = summarize_ranked_task(
            task=task,
            ranked_doc_ids_by_query=exact_doc_ids,
        )

    stats = _measurement_stats(durations)
    candidate_counts = [len(row) for row in first_candidate_positions]
    query_count = len(query_matrices)
    batch_mean_seconds = stats["mean_seconds"]
    return SnapshotTachiomBenchmarkSummary(
        dataset_id=str(manifest["dataset_id"]),
        model_name=str(manifest["model_name"]),
        primary_metric=task_metrics.primary_metric,
        primary_value=task_metrics.primary_value,
        mean_ndcg_at_k=task_metrics.mean_ndcg_at_k,
        mean_reciprocal_rank=task_metrics.mean_reciprocal_rank,
        mean_recall_at_k=task_metrics.mean_recall_at_k,
        success_rate_at_k=task_metrics.success_rate_at_k,
        exact_primary_value=None if exact_metrics is None else exact_metrics.primary_value,
        exact_mean_ndcg_at_k=(
            None if exact_metrics is None else exact_metrics.mean_ndcg_at_k
        ),
        exact_mean_recall_at_k=(
            None if exact_metrics is None else exact_metrics.mean_recall_at_k
        ),
        candidate_recall_at_k_vs_exact=(
            None
            if reference_positions is None
            else mean_candidate_set_recall_at_k(
                candidate_positions_by_query=first_candidate_positions,
                reference_positions_by_query=reference_positions,
                k=final_k,
            )
        ),
        final_recall_at_k_vs_exact=(
            None
            if reference_positions is None
            else mean_recall_at_k(
                candidate_positions_by_query=first_final_positions,
                reference_positions_by_query=reference_positions,
                k=final_k,
            )
        ),
        query_batch_mean_seconds=batch_mean_seconds,
        query_mean_seconds=batch_mean_seconds / float(query_count),
        query_qps=float(query_count) / batch_mean_seconds,
        build_seconds=build_seconds,
        k=final_k,
        query_count=query_count,
        document_count=documents.loaded_document_count,
        document_vector_count_total=documents.loaded_vector_count,
        document_vector_count_mean=round(
            documents.loaded_vector_count / float(documents.loaded_document_count)
        ),
        query_vector_count_total=queries.query_vector_count,
        query_vector_count_mean=round(queries.query_vector_count / float(query_count)),
        vector_dim=int(manifest["vector_dim"]),
        engine=engine,
        index_kind=index.index_kind,
        vector_metric="dot_product",
        source_vector_dtype=documents.source_vector_dtype,
        source_token_id_dtype=documents.source_token_id_dtype,
        loaded_payload_bytes=documents.loaded_payload_bytes,
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
        epistemic_status={
            "claim": (
                "Tachiom built directly from a sharded encoded snapshot. This "
                "avoids JSON and Python vector-list materialization for the "
                "selected snapshot rows."
            ),
            "remaining_paper_scale_blocker": (
                "The current TAC reference still loads the selected vector "
                "matrix into memory. Full MS MARCO needs a streaming TAC/PQ "
                "builder over snapshot shards."
            ),
        },
    )


def _build_index(
    documents: PackedSnapshotDocuments,
    *,
    engine: str,
    tac_config: TachiomTacConfig,
    hnsw_config: TachiomHnswConfig | None,
    pq_config: TachiomResidualPqConfig | None,
    final_k: int,
) -> object:
    tac_index = TachiomTacIndex.build_packed(
        doc_ids=documents.doc_ids,
        doc_offsets=documents.doc_offsets,
        token_vectors=documents.token_vectors,
        token_ids=documents.token_ids,
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
    raise ValueError(f"unsupported snapshot Tachiom engine: {engine}")


def _search_snapshot_positions(
    index: object,
    query_matrices: Sequence[np.ndarray],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        index.search_batch_positions(query[None, :, :], final_k=final_k)[0]
        for query in query_matrices
    )


def _candidate_snapshot_positions(
    index: object,
    query_matrices: Sequence[np.ndarray],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        index.candidate_positions_batch(query[None, :, :], final_k=final_k)[0]
        for query in query_matrices
    )


def _exact_search_snapshot_positions(
    *,
    documents: PackedSnapshotDocuments,
    queries: Sequence[np.ndarray],
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for query in queries:
        scores = np.empty(documents.loaded_document_count, dtype=VECTOR_DTYPE)
        for doc_index in range(documents.loaded_document_count):
            start = int(documents.doc_offsets[doc_index])
            stop = int(documents.doc_offsets[doc_index + 1])
            document = documents.token_vectors[start:stop]
            interactions = np.matmul(query, document.T)
            scores[doc_index] = np.max(interactions, axis=1).sum()
        positions = np.arange(documents.loaded_document_count)
        order = np.lexsort((positions, -scores))
        rows.append(tuple(int(position) for position in order[:final_k]))
    return tuple(rows)


def _positions_to_doc_id_rows(
    positions_by_query: Sequence[Sequence[int]],
    *,
    doc_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(str(doc_ids[int(position)]) for position in row)
        for row in positions_by_query
    )


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
