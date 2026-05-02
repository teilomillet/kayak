"""Benchmarks query-time search over materialized streaming TAC/PQ artifacts.

This module owns the evidence path after streaming construction: load the
on-disk index, search snapshot query sidecars, and optionally compare against
exact snapshot search for bounded slices. It does not build the index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .encoded_snapshot_loader import (
    SnapshotQueries,
    load_packed_snapshot_documents,
    load_snapshot_manifest,
    load_snapshot_queries,
)
from .dtypes import VECTOR_DTYPE
from .judged_metrics import TaskMetricSummary, summarize_ranked_task
from .tachiom_metrics import mean_candidate_set_recall_at_k, mean_recall_at_k
from .tachiom_snapshot_benchmark import _exact_search_snapshot_positions
from .tachiom_streaming_search import (
    load_streaming_tachiom_hnsw_pq_mojo_address_index,
    load_streaming_tachiom_hnsw_pq_mojo_index,
    load_streaming_tachiom_hnsw_pq_index,
    load_streaming_tachiom_pq_mojo_index,
    load_streaming_tachiom_pq_index,
)


@dataclass(frozen=True, slots=True)
class StreamingTachiomBenchmarkSummary:
    dataset_id: str
    model_name: str
    primary_metric: str
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    exact_primary_value: float | None
    candidate_recall_at_k_vs_exact: float | None
    final_recall_at_k_vs_exact: float | None
    query_batch_mean_seconds: float
    query_mean_seconds: float
    query_qps: float
    max_query_batch_size: int | None
    candidate_pruning_alpha: float | None
    candidate_pruning_alpha_source: str
    k: int
    query_count: int
    document_count: int
    document_vector_count_total: int
    query_vector_count_total: int
    query_vector_count_mean: int
    vector_dim: int
    engine: str
    index_kind: str
    rerank_kind: str
    index_bytes: int
    centroid_count: int
    posting_count: int
    candidate_window_min_count: int
    candidate_window_mean_count: float
    candidate_window_max_count: int
    query_warmup_iterations: int
    query_measurement_iterations: int
    epistemic_status: dict[str, object]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StreamingTachiomExactReference:
    """Exact MaxSim reference rankings for one streaming snapshot/index prefix."""

    reference_positions: tuple[tuple[int, ...], ...]
    exact_metrics: TaskMetricSummary
    doc_ids: tuple[str, ...]
    document_count: int
    document_vector_count_total: int
    query_count: int
    query_vector_count_total: int
    k: int
    loaded_payload_bytes: int

    def to_json_ready(self) -> dict[str, object]:
        return {
            "exact_metrics": asdict(self.exact_metrics),
            "document_count": self.document_count,
            "document_vector_count_total": self.document_vector_count_total,
            "query_count": self.query_count,
            "query_vector_count_total": self.query_vector_count_total,
            "k": self.k,
            "loaded_payload_bytes": self.loaded_payload_bytes,
        }


def benchmark_streaming_tachiom_index(
    *,
    snapshot_root: Path,
    index_root: Path,
    query_limit: int | None = None,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
    run_exact: bool = False,
    max_exact_vector_count: int | None = None,
    engine: str = "streaming_tac_pq",
    graph_root: Path | None = None,
    max_query_batch_size: int | None = None,
    candidate_pruning_alpha: float | None = None,
    disable_candidate_pruning: bool = False,
    exact_reference: StreamingTachiomExactReference | None = None,
) -> StreamingTachiomBenchmarkSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    if max_query_batch_size is not None and max_query_batch_size <= 0:
        raise ValueError("max_query_batch_size must be positive when provided")
    if disable_candidate_pruning and candidate_pruning_alpha is not None:
        raise ValueError(
            "candidate_pruning_alpha and disable_candidate_pruning are mutually exclusive"
        )
    _validate_candidate_pruning_alpha(candidate_pruning_alpha)

    snapshot_manifest = load_snapshot_manifest(snapshot_root)
    index = _load_streaming_index(
        index_root=index_root,
        engine=engine,
        graph_root=graph_root,
    )
    candidate_pruning_alpha_source = "artifact"
    if disable_candidate_pruning:
        index = _with_candidate_pruning_alpha(index, None)
        candidate_pruning_alpha_source = "disabled_override"
    elif candidate_pruning_alpha is not None:
        index = _with_candidate_pruning_alpha(index, candidate_pruning_alpha)
        candidate_pruning_alpha_source = "override"
    effective_candidate_pruning_alpha = _candidate_pruning_alpha_for_index(index)
    queries = load_snapshot_queries(snapshot_root, query_limit=query_limit)
    final_k = queries.k

    for _ in range(warmup_iterations):
        _search_positions(
            index,
            queries.query_matrices,
            final_k=final_k,
            max_query_batch_size=max_query_batch_size,
        )

    durations: list[float] = []
    first_final_positions: tuple[tuple[int, ...], ...] | None = None
    first_candidate_positions: tuple[tuple[int, ...], ...] | None = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        final_positions = _search_positions(
            index,
            queries.query_matrices,
            final_k=final_k,
            max_query_batch_size=max_query_batch_size,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_final_positions = final_positions
            first_candidate_positions = _candidate_positions(
                index,
                queries.query_matrices,
                final_k=final_k,
                max_query_batch_size=max_query_batch_size,
            )
    if first_final_positions is None or first_candidate_positions is None:
        raise RuntimeError("streaming benchmark produced no rankings")

    task = _streaming_task_metadata(
        snapshot_manifest=snapshot_manifest,
        index_doc_ids=index.doc_ids,
        queries=queries,
    )
    ranked_doc_ids = _positions_to_doc_id_rows(
        first_final_positions,
        doc_ids=index.doc_ids,
    )
    task_metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids,
    )

    exact_reference_source = None
    if exact_reference is not None:
        run_exact = True
        exact_reference_source = "provided"
        _validate_exact_reference(
            exact_reference,
            index=index,
            queries=queries,
            final_k=final_k,
        )
    elif run_exact:
        exact_reference_source = "computed"
        exact_reference = _compute_streaming_exact_reference(
            snapshot_root=snapshot_root,
            index=index,
            queries=queries,
            task=task,
            max_exact_vector_count=max_exact_vector_count,
        )
    reference_positions = (
        None if exact_reference is None else exact_reference.reference_positions
    )
    exact_metrics = None if exact_reference is None else exact_reference.exact_metrics

    stats = _measurement_stats(durations)
    query_count = len(queries.query_matrices)
    candidate_counts = [len(row) for row in first_candidate_positions]
    batch_mean_seconds = stats["mean_seconds"]
    return StreamingTachiomBenchmarkSummary(
        dataset_id=str(snapshot_manifest["dataset_id"]),
        model_name=str(snapshot_manifest["model_name"]),
        primary_metric=task_metrics.primary_metric,
        primary_value=task_metrics.primary_value,
        mean_ndcg_at_k=task_metrics.mean_ndcg_at_k,
        mean_reciprocal_rank=task_metrics.mean_reciprocal_rank,
        mean_recall_at_k=task_metrics.mean_recall_at_k,
        success_rate_at_k=task_metrics.success_rate_at_k,
        exact_primary_value=None if exact_metrics is None else exact_metrics.primary_value,
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
        max_query_batch_size=max_query_batch_size,
        candidate_pruning_alpha=effective_candidate_pruning_alpha,
        candidate_pruning_alpha_source=candidate_pruning_alpha_source,
        k=final_k,
        query_count=query_count,
        document_count=index.document_count,
        document_vector_count_total=index.total_vector_count,
        query_vector_count_total=queries.query_vector_count,
        query_vector_count_mean=round(queries.query_vector_count / float(query_count)),
        vector_dim=index.vector_dim,
        engine=engine,
        index_kind=index.index_kind,
        rerank_kind=index.rerank_kind,
        index_bytes=index.index_bytes,
        centroid_count=index.centroid_count,
        posting_count=index.posting_count,
        candidate_window_min_count=min(candidate_counts),
        candidate_window_mean_count=float(sum(candidate_counts) / len(candidate_counts)),
        candidate_window_max_count=max(candidate_counts),
        query_warmup_iterations=warmup_iterations,
        query_measurement_iterations=measurement_iterations,
        epistemic_status={
            "claim": (
                "Search loaded the materialized streaming TAC/PQ artifact with "
                "memory-mapped query-time arrays."
            ),
            "remaining_paper_scale_blocker": (
                _remaining_blocker_for_engine(engine)
            ),
            "candidate_pruning_alpha_source": candidate_pruning_alpha_source,
            "exact_reference_source": exact_reference_source,
        },
    )


def build_streaming_tachiom_exact_reference(
    *,
    snapshot_root: Path,
    index_root: Path,
    query_limit: int | None = None,
    max_exact_vector_count: int | None = None,
) -> StreamingTachiomExactReference:
    """Build exact MaxSim rankings once for sweeps over the same artifact."""

    snapshot_manifest = load_snapshot_manifest(snapshot_root)
    index = load_streaming_tachiom_pq_index(index_root)
    queries = load_snapshot_queries(snapshot_root, query_limit=query_limit)
    task = _streaming_task_metadata(
        snapshot_manifest=snapshot_manifest,
        index_doc_ids=index.doc_ids,
        queries=queries,
    )
    return _compute_streaming_exact_reference(
        snapshot_root=snapshot_root,
        index=index,
        queries=queries,
        task=task,
        max_exact_vector_count=max_exact_vector_count,
    )


def _compute_streaming_exact_reference(
    *,
    snapshot_root: Path,
    index: object,
    queries: SnapshotQueries,
    task: dict[str, object],
    max_exact_vector_count: int | None,
) -> StreamingTachiomExactReference:
    documents = load_packed_snapshot_documents(
        snapshot_root,
        document_limit=index.document_count,
        max_vector_count=max_exact_vector_count,
    )
    if tuple(documents.doc_ids) != tuple(index.doc_ids):
        raise ValueError("snapshot document prefix does not match streaming index doc ids")
    reference_positions = _exact_search_snapshot_positions(
        documents=documents,
        queries=queries.query_matrices,
        final_k=queries.k,
    )
    exact_doc_ids = _positions_to_doc_id_rows(
        reference_positions,
        doc_ids=index.doc_ids,
    )
    exact_metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=exact_doc_ids,
    )
    return StreamingTachiomExactReference(
        reference_positions=reference_positions,
        exact_metrics=exact_metrics,
        doc_ids=tuple(index.doc_ids),
        document_count=index.document_count,
        document_vector_count_total=index.total_vector_count,
        query_count=len(queries.query_matrices),
        query_vector_count_total=queries.query_vector_count,
        k=queries.k,
        loaded_payload_bytes=documents.loaded_payload_bytes,
    )


def _validate_exact_reference(
    exact_reference: StreamingTachiomExactReference,
    *,
    index: object,
    queries: SnapshotQueries,
    final_k: int,
) -> None:
    if exact_reference.doc_ids != tuple(index.doc_ids):
        raise ValueError("exact reference doc ids do not match streaming index")
    if exact_reference.query_count != len(queries.query_matrices):
        raise ValueError("exact reference query count does not match benchmark queries")
    if exact_reference.query_vector_count_total != queries.query_vector_count:
        raise ValueError("exact reference query vector count does not match queries")
    if exact_reference.k != final_k:
        raise ValueError("exact reference k does not match benchmark k")


def _validate_candidate_pruning_alpha(candidate_pruning_alpha: float | None) -> None:
    if candidate_pruning_alpha is None:
        return
    if not (0.0 < candidate_pruning_alpha < 1.0):
        raise ValueError("candidate_pruning_alpha must be between 0 and 1")


def _with_candidate_pruning_alpha(
    index: object,
    candidate_pruning_alpha: float | None,
) -> object:
    """Return an index view with a query-time pruning override.

    Prepared Mojo readers keep their native index handle because pruning alpha is
    passed per search call; only the lightweight Python metadata wrapper changes.
    """

    _validate_candidate_pruning_alpha(candidate_pruning_alpha)
    base_index = getattr(index, "base_index", None)
    if base_index is not None and hasattr(base_index, "candidate_pruning_alpha"):
        return replace(
            index,
            base_index=replace(
                base_index,
                candidate_pruning_alpha=candidate_pruning_alpha,
            ),
        )
    if hasattr(index, "candidate_pruning_alpha"):
        return replace(index, candidate_pruning_alpha=candidate_pruning_alpha)
    raise TypeError("streaming index does not expose candidate_pruning_alpha")


def _candidate_pruning_alpha_for_index(index: object) -> float | None:
    base_index = getattr(index, "base_index", index)
    return base_index.candidate_pruning_alpha


def _load_streaming_index(
    *,
    index_root: Path,
    engine: str,
    graph_root: Path | None,
) -> object:
    if engine == "streaming_tac_pq":
        return load_streaming_tachiom_pq_index(index_root)
    if engine == "streaming_tac_pq_mojo":
        return load_streaming_tachiom_pq_mojo_index(index_root)
    if engine == "streaming_tac_hnsw_pq":
        return load_streaming_tachiom_hnsw_pq_index(
            index_root,
            graph_root=graph_root,
        )
    if engine == "streaming_tac_hnsw_pq_mojo":
        return load_streaming_tachiom_hnsw_pq_mojo_index(
            index_root,
            graph_root=graph_root,
        )
    if engine == "streaming_tac_hnsw_pq_mojo_address":
        return load_streaming_tachiom_hnsw_pq_mojo_address_index(
            index_root,
            graph_root=graph_root,
        )
    raise ValueError(f"unsupported streaming Tachiom engine: {engine}")


def _remaining_blocker_for_engine(engine: str) -> str:
    if engine == "streaming_tac_pq_mojo":
        return (
            "This path validates native Mojo TAC/PQ search over the streaming "
            "artifact, but it still uses exact centroid scan rather than the "
            "paper's HNSW centroid traversal."
        )
    if engine == "streaming_tac_hnsw_pq":
        return (
            "This path validates persisted HNSW centroid traversal, but graph "
            "construction is still the Python reference and has not been "
            "validated at paper-scale centroid counts."
        )
    if engine == "streaming_tac_hnsw_pq_mojo":
        return (
            "This path validates native HNSW candidate generation plus sparse "
            "residual-PQ rerank over the streaming artifact. Graph construction "
            "is still the Python reference and full paper-scale corpus evidence "
            "has not been measured here."
        )
    if engine == "streaming_tac_hnsw_pq_mojo_address":
        return (
            "This path validates address-backed native HNSW candidate generation "
            "plus sparse residual-PQ rerank over the streaming artifact. It avoids "
            "Python-list materialization of index arrays, but graph construction "
            "is still the Python reference and full paper-scale corpus evidence "
            "has not been measured here."
        )
    return (
        "This reader uses exact centroid scan, not the paper's HNSW centroid "
        "traversal. It validates artifact searchability, not paper throughput."
    )


def _streaming_task_metadata(
    *,
    snapshot_manifest: dict[str, object],
    index_doc_ids: Sequence[str],
    queries: SnapshotQueries,
) -> dict[str, object]:
    return {
        "dataset_id": str(snapshot_manifest["dataset_id"]),
        "model_name": str(snapshot_manifest["model_name"]),
        "family": "msmarco",
        "slice_name": (
            f"streaming_index_docs_{len(index_doc_ids)}_"
            f"queries_{len(queries.query_ids)}"
        ),
        "primary_metric": queries.primary_metric,
        "k": queries.k,
        "documents": [{"doc_id": doc_id} for doc_id in index_doc_ids],
        "queries": [
            {
                "query_id": query_id,
                "text": text,
                "relevant_doc_ids": list(relevant_doc_ids),
            }
            for query_id, text, relevant_doc_ids in zip(
                queries.query_ids,
                queries.texts,
                queries.relevant_doc_ids,
                strict=True,
            )
        ],
    }


def _search_positions(
    index: object,
    query_matrices: Sequence[np.ndarray],
    *,
    final_k: int,
    max_query_batch_size: int | None,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...] | None] = [None] * len(query_matrices)
    for group in _same_shape_query_groups(
        query_matrices,
        max_query_batch_size=max_query_batch_size,
    ):
        batch = np.ascontiguousarray(
            np.stack([query_matrices[index] for index in group]),
            dtype=VECTOR_DTYPE,
        )
        batch_rows = index.search_batch_positions(batch, final_k=final_k)
        for row_index, positions in zip(group, batch_rows, strict=True):
            rows[row_index] = tuple(positions)
    return tuple(_require_row(row) for row in rows)


def _candidate_positions(
    index: object,
    query_matrices: Sequence[np.ndarray],
    *,
    final_k: int,
    max_query_batch_size: int | None,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...] | None] = [None] * len(query_matrices)
    for group in _same_shape_query_groups(
        query_matrices,
        max_query_batch_size=max_query_batch_size,
    ):
        batch = np.ascontiguousarray(
            np.stack([query_matrices[index] for index in group]),
            dtype=VECTOR_DTYPE,
        )
        batch_rows = index.candidate_positions_batch(batch, final_k=final_k)
        for row_index, positions in zip(group, batch_rows, strict=True):
            rows[row_index] = tuple(positions)
    return tuple(_require_row(row) for row in rows)


def _same_shape_query_groups(
    query_matrices: Sequence[np.ndarray],
    *,
    max_query_batch_size: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    if max_query_batch_size is not None and max_query_batch_size <= 0:
        raise ValueError("max_query_batch_size must be positive when provided")
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, query in enumerate(query_matrices):
        groups.setdefault(tuple(int(axis) for axis in query.shape), []).append(index)
    if max_query_batch_size is None:
        return tuple(tuple(indices) for indices in groups.values())
    return tuple(
        tuple(indices[start : start + max_query_batch_size])
        for indices in groups.values()
        for start in range(0, len(indices), max_query_batch_size)
    )


def _require_row(row: tuple[int, ...] | None) -> tuple[int, ...]:
    if row is None:
        raise RuntimeError("batched search did not fill every query row")
    return row


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
