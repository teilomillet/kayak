"""Owns candidate-window sweeps for the reference LEMUR benchmark path.

This module keeps the LEMUR question narrow and measurable:
- fit one latent reference model per latent dimension
- benchmark shortlist-plus-rerank quality across candidate windows
- compare search and shortlist faithfulness against an exact baseline

It does not claim ANN-backed serving or trained-MLP reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping, Sequence

import kayak

from .kayak_task_benchmark import (
    benchmark_queries_with_kayak_exact,
    rank_task_with_kayak_exact,
)
from .lemur_task_benchmark import (
    benchmark_queries_with_reference_lemur_model,
)
from .reference_lemur import fit_reference_lemur


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


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _shortlist_doc_ids_by_query(
    *,
    queries: Sequence[kayak.LateQuery],
    model: Any,
    candidate_k: int,
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(hit.doc_id for hit in model.similarity_scores(query).topk(candidate_k))
        for query in queries
    )


def _mean_exact_topk_shortlist_recall(
    *,
    exact_ranked_doc_ids_by_query: Sequence[Sequence[str]],
    shortlist_doc_ids_by_query: Sequence[Sequence[str]],
) -> float:
    if len(exact_ranked_doc_ids_by_query) != len(shortlist_doc_ids_by_query):
        raise ValueError("exact and shortlist rankings must align by query")
    if not exact_ranked_doc_ids_by_query:
        raise ValueError("shortlist recall requires at least one query")

    recall_total = 0.0
    for exact_ranked_doc_ids, shortlist_doc_ids in zip(
        exact_ranked_doc_ids_by_query,
        shortlist_doc_ids_by_query,
        strict=True,
    ):
        if not exact_ranked_doc_ids:
            recall_total += 0.0
            continue
        shortlist = set(shortlist_doc_ids)
        hit_count = 0
        for doc_id in exact_ranked_doc_ids:
            if doc_id in shortlist:
                hit_count += 1
        recall_total += hit_count / float(len(exact_ranked_doc_ids))

    return recall_total / float(len(exact_ranked_doc_ids_by_query))


@dataclass(frozen=True, slots=True)
class LemurCandidateSweepRow:
    latent_dim: int
    candidate_k: int
    landmark_count: int
    fit_seconds: float
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    mean_exact_topk_shortlist_recall: float
    mean_stage1_seconds: float
    mean_rerank_seconds: float
    mean_search_seconds: float
    primary_ratio_vs_exact: float | None
    mean_search_seconds_ratio_vs_exact: float | None


@dataclass(frozen=True, slots=True)
class LemurCandidateSweepSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    document_count: int
    vector_dim: int
    activation: str
    query_divisor: float
    apply_layer_norm: bool
    exact_backend: str
    rerank_backend: str
    exact_primary_value: float
    exact_mean_search_seconds: float
    rows: tuple[LemurCandidateSweepRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_reference_lemur_candidate_sweep(
    task: Mapping[str, Any],
    *,
    latent_dims: Sequence[int],
    candidate_ks: Sequence[int],
    activation: str = "gelu",
    query_divisor: float = 32.0,
    landmark_count: int | None = None,
    feature_weights: Any | None = None,
    landmark_vectors: Any | None = None,
    apply_layer_norm: bool = True,
    layer_norm_eps: float = 1e-5,
    pinv_rcond: float = 1e-6,
    seed: int = 0,
    exact_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
    rerank_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
) -> LemurCandidateSweepSummary:
    if not latent_dims:
        raise ValueError("latent_dims must not be empty")
    if not candidate_ks:
        raise ValueError("candidate_ks must not be empty")

    index = _build_index(task)
    queries = _build_queries(task)
    exact_summary = benchmark_queries_with_kayak_exact(
        task=task,
        index=index,
        queries=queries,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        backend=exact_backend,
    )
    exact_ranked_doc_ids_by_query = rank_task_with_kayak_exact(
        task,
        backend=exact_backend,
    )

    rows: list[LemurCandidateSweepRow] = []
    resolved_latent_dims = sorted({int(latent_dim) for latent_dim in latent_dims})
    resolved_candidate_ks = sorted({int(candidate_k) for candidate_k in candidate_ks})
    for latent_dim in resolved_latent_dims:
        fit_start = time.perf_counter()
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
        fit_seconds = time.perf_counter() - fit_start

        for candidate_k in resolved_candidate_ks:
            if candidate_k < int(task["k"]):
                raise ValueError("candidate_k must be greater than or equal to final_k")

            summary = benchmark_queries_with_reference_lemur_model(
                task=task,
                index=index,
                queries=queries,
                model=model,
                candidate_k=candidate_k,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                rerank_backend=rerank_backend,
            )
            shortlist_doc_ids_by_query = _shortlist_doc_ids_by_query(
                queries=queries,
                model=model,
                candidate_k=candidate_k,
            )
            rows.append(
                LemurCandidateSweepRow(
                    latent_dim=latent_dim,
                    candidate_k=candidate_k,
                    landmark_count=model.landmark_count,
                    fit_seconds=fit_seconds,
                    primary_value=summary.primary_value,
                    mean_ndcg_at_k=summary.mean_ndcg_at_k,
                    mean_reciprocal_rank=summary.mean_reciprocal_rank,
                    mean_recall_at_k=summary.mean_recall_at_k,
                    success_rate_at_k=summary.success_rate_at_k,
                    mean_exact_topk_shortlist_recall=_mean_exact_topk_shortlist_recall(
                        exact_ranked_doc_ids_by_query=exact_ranked_doc_ids_by_query,
                        shortlist_doc_ids_by_query=shortlist_doc_ids_by_query,
                    ),
                    mean_stage1_seconds=summary.mean_stage1_seconds,
                    mean_rerank_seconds=summary.mean_rerank_seconds,
                    mean_search_seconds=summary.mean_search_seconds,
                    primary_ratio_vs_exact=_safe_ratio(
                        summary.primary_value,
                        exact_summary.primary_value,
                    ),
                    mean_search_seconds_ratio_vs_exact=_safe_ratio(
                        summary.mean_search_seconds,
                        exact_summary.mean_search_seconds,
                    ),
                )
            )

    return LemurCandidateSweepSummary(
        dataset_id=str(task["dataset_id"]),
        model_name=str(task["model_name"]),
        family=str(task["family"]),
        slice_name=str(task["slice_name"]),
        primary_metric=str(task["primary_metric"]),
        k=int(task["k"]),
        document_count=len(task["documents"]),
        vector_dim=int(task["vector_dim"]),
        activation=activation,
        query_divisor=float(query_divisor),
        apply_layer_norm=apply_layer_norm,
        exact_backend=exact_backend,
        rerank_backend=rerank_backend,
        exact_primary_value=exact_summary.primary_value,
        exact_mean_search_seconds=exact_summary.mean_search_seconds,
        rows=tuple(rows),
    )
