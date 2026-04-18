"""Owns Python-driven LEMUR-shaped task benchmarks for encoded task JSON.

This module keeps the learned-reduction benchmark explicit:
- fit a reference LEMUR model on the task corpus
- generate a latent shortlist with the fitted model
- exact-rerank that shortlist with Kayak MaxSim

It does not claim native ANN execution or a fully trained paper reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping

import kayak

from .judged_metrics import summarize_ranked_task
from .reference_lemur import LemurReferenceModel, fit_reference_lemur


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


def _rank_one_query(
    *,
    query: kayak.LateQuery,
    index: kayak.LateIndex,
    model: LemurReferenceModel,
    candidate_k: int,
    final_k: int,
    rerank_backend: str,
) -> tuple[tuple[str, ...], float, float]:
    stage1_start = time.perf_counter()
    shortlist = tuple(
        hit.doc_id
        for hit in model.similarity_scores(query).topk(candidate_k)
    )
    stage1_seconds = time.perf_counter() - stage1_start

    rerank_start = time.perf_counter()
    reranked = kayak.maxsim(
        query,
        index.select(shortlist),
        backend=rerank_backend,
    ).topk(final_k)
    rerank_seconds = time.perf_counter() - rerank_start
    return (
        tuple(hit.doc_id for hit in reranked),
        stage1_seconds,
        rerank_seconds,
    )


def rank_task_with_reference_lemur(
    task: Mapping[str, Any],
    *,
    latent_dim: int,
    candidate_k: int,
    activation: str = "gelu",
    query_divisor: float = 32.0,
    landmark_count: int | None = None,
    feature_weights: Any | None = None,
    landmark_vectors: Any | None = None,
    apply_layer_norm: bool = True,
    layer_norm_eps: float = 1e-5,
    pinv_rcond: float = 1e-6,
    seed: int = 0,
    final_k: int | None = None,
    rerank_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
) -> tuple[tuple[str, ...], ...]:
    index = _build_index(task)
    queries = _build_queries(task)
    resolved_final_k = int(task["k"]) if final_k is None else final_k
    if candidate_k < resolved_final_k:
        raise ValueError("candidate_k must be greater than or equal to final_k")

    model = fit_reference_lemur(
        index,
        latent_dim=latent_dim,
        activation=activation,
        query_divisor=query_divisor,
        landmark_count=landmark_count,
        feature_weights=feature_weights,
        landmark_vectors=landmark_vectors,
        apply_layer_norm=apply_layer_norm,
        layer_norm_eps=layer_norm_eps,
        pinv_rcond=pinv_rcond,
        seed=seed,
    )
    return tuple(
        _rank_one_query(
            query=query,
            index=index,
            model=model,
            candidate_k=candidate_k,
            final_k=resolved_final_k,
            rerank_backend=rerank_backend,
        )[0]
        for query in queries
    )


@dataclass(frozen=True, slots=True)
class LemurTaskBenchmarkSummary:
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
    mean_stage1_seconds: float
    mean_rerank_seconds: float
    mean_search_seconds: float
    k: int
    candidate_k: int
    query_count: int
    document_count: int
    vector_dim: int
    latent_dim: int
    landmark_count: int
    activation: str
    query_divisor: float
    apply_layer_norm: bool
    engine: str
    index_kind: str
    stage1_backend: str
    rerank_backend: str
    query_warmup_iterations: int
    query_measurement_iterations: int

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_task_with_reference_lemur(
    task: Mapping[str, Any],
    *,
    latent_dim: int,
    candidate_k: int,
    activation: str = "gelu",
    query_divisor: float = 32.0,
    landmark_count: int | None = None,
    feature_weights: Any | None = None,
    landmark_vectors: Any | None = None,
    apply_layer_norm: bool = True,
    layer_norm_eps: float = 1e-5,
    pinv_rcond: float = 1e-6,
    seed: int = 0,
    final_k: int | None = None,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
    rerank_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
) -> LemurTaskBenchmarkSummary:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    index = _build_index(task)
    queries = _build_queries(task)
    resolved_final_k = int(task["k"]) if final_k is None else final_k
    if candidate_k < resolved_final_k:
        raise ValueError("candidate_k must be greater than or equal to final_k")

    model = fit_reference_lemur(
        index,
        latent_dim=latent_dim,
        activation=activation,
        query_divisor=query_divisor,
        landmark_count=landmark_count,
        feature_weights=feature_weights,
        landmark_vectors=landmark_vectors,
        apply_layer_norm=apply_layer_norm,
        layer_norm_eps=layer_norm_eps,
        pinv_rcond=pinv_rcond,
        seed=seed,
    )

    for _ in range(warmup_iterations):
        for query in queries:
            _rank_one_query(
                query=query,
                index=index,
                model=model,
                candidate_k=candidate_k,
                final_k=resolved_final_k,
                rerank_backend=rerank_backend,
            )

    stage1_elapsed_seconds: list[float] = []
    rerank_elapsed_seconds: list[float] = []
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for measurement_iteration in range(measurement_iterations):
        current_rankings: list[tuple[str, ...]] = []
        current_stage1_total = 0.0
        current_rerank_total = 0.0
        for query in queries:
            ranked, stage1_seconds, rerank_seconds = _rank_one_query(
                query=query,
                index=index,
                model=model,
                candidate_k=candidate_k,
                final_k=resolved_final_k,
                rerank_backend=rerank_backend,
            )
            current_rankings.append(ranked)
            current_stage1_total += stage1_seconds
            current_rerank_total += rerank_seconds

        stage1_elapsed_seconds.append(current_stage1_total / float(len(queries)))
        rerank_elapsed_seconds.append(current_rerank_total / float(len(queries)))
        if measurement_iteration == 0:
            ranked_doc_ids_by_query = current_rankings

    task_metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    return LemurTaskBenchmarkSummary(
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
        mean_stage1_seconds=float(
            sum(stage1_elapsed_seconds) / len(stage1_elapsed_seconds)
        ),
        mean_rerank_seconds=float(
            sum(rerank_elapsed_seconds) / len(rerank_elapsed_seconds)
        ),
        mean_search_seconds=float(
            (sum(stage1_elapsed_seconds) + sum(rerank_elapsed_seconds))
            / len(stage1_elapsed_seconds)
        ),
        k=task_metrics.k,
        candidate_k=candidate_k,
        query_count=task_metrics.query_count,
        document_count=task_metrics.document_count,
        vector_dim=int(task["vector_dim"]),
        latent_dim=model.latent_dim,
        landmark_count=model.landmark_count,
        activation=model.activation,
        query_divisor=model.query_divisor,
        apply_layer_norm=model.apply_layer_norm,
        engine="lemur_reference",
        index_kind="latent_single_vector_reference",
        stage1_backend="lemur_reference",
        rerank_backend=rerank_backend,
        query_warmup_iterations=warmup_iterations,
        query_measurement_iterations=measurement_iterations,
    )
