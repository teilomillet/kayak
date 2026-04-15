"""Owns reproducible R2MED/Biology policy-expansion sweeps.

This module owns:
- full-index late-interaction fusion over public R2MED query rewrites
- shortlist late-interaction rescoring for seed-policy expansions
- explicit reporting of the best maxsim-only and rescored expansion policies

This module does not own:
- corpus snapshot building
- non-late-interaction rerankers
- public leaderboard scraping
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import kayak
import numpy as np
from kayak.stores import DirectoryLateStore

from .colbert_encoder import DEFAULT_MODEL_NAME
from .judged_metrics import summarize_ranked_task
from .prepared_index_cache import clear_prepared_packed_index_cache
from .r2med_biology_full import DEFAULT_DATASET_ID, DEFAULT_SNAPSHOT_ROOT
from .r2med_biology_late_interaction_quality import (
    DEFAULT_VARIANT_CACHE_ROOT,
    LateInteractionPolicySpec,
)
from .r2med_biology_query_variants import (
    R2MED_QUERY_VARIANTS,
    default_r2med_query_variant,
    ensure_r2med_query_variant_cache,
)
from .reference_topk_pooling import topk_mean_similarity_scores
from .score_fusion import weighted_sum_score_batches


@dataclass(frozen=True, slots=True)
class PolicyExpansionRow:
    policy_name: str
    weights_by_variant_name: dict[str, float]
    mean_ndcg_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    success_rate_at_k: float
    expansion_variant_name: str | None = None
    expansion_weight: float | None = None
    shortlist_materialization_seconds: float | None = None
    rescoring_seconds: float | None = None
    rescoring_seconds_per_query: float | None = None

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class R2MEDPolicyExpansionSummary:
    dataset_id: str
    model_name: str
    document_count: int
    query_count: int
    vector_dim: int
    k: int
    backend: str
    shortlist_k: int
    match_k: int
    seed_policy_name: str
    seed_weights_by_variant_name: dict[str, float]
    rescored_variant_name: str
    variant_score_seconds: dict[str, float]
    single_variant_rows: tuple[PolicyExpansionRow, ...]
    maxsim_expansion_rows: tuple[PolicyExpansionRow, ...]
    rescored_expansion_rows: tuple[PolicyExpansionRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "document_count": self.document_count,
            "query_count": self.query_count,
            "vector_dim": self.vector_dim,
            "k": self.k,
            "backend": self.backend,
            "shortlist_k": self.shortlist_k,
            "match_k": self.match_k,
            "seed_policy_name": self.seed_policy_name,
            "seed_weights_by_variant_name": dict(self.seed_weights_by_variant_name),
            "rescored_variant_name": self.rescored_variant_name,
            "variant_score_seconds": dict(self.variant_score_seconds),
            "single_variant_rows": [
                row.to_json_ready() for row in self.single_variant_rows
            ],
            "maxsim_expansion_rows": [
                row.to_json_ready() for row in self.maxsim_expansion_rows
            ],
            "rescored_expansion_rows": [
                row.to_json_ready() for row in self.rescored_expansion_rows
            ],
        }


def default_seed_policy() -> LateInteractionPolicySpec:
    return LateInteractionPolicySpec(
        "query2doc_plus_1.9_lamer",
        {"query2doc_gpt4": 1.0, "lamer_gpt4": 1.9},
    )


def default_expansion_variant_names() -> tuple[str, ...]:
    seed_variant_names = set(default_seed_policy().weights_by_variant_name)
    return tuple(
        spec.name
        for spec in R2MED_QUERY_VARIANTS
        if spec.name not in seed_variant_names
    )


def default_expansion_weights() -> tuple[float, ...]:
    return (
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.75,
        1.0,
        1.25,
    )


def _variant_cache_path(cache_root: Path, variant_name: str) -> Path:
    return cache_root / f"{variant_name}.json"


def _task_rows(cache: object, *, document_count: int) -> tuple[dict[str, object], dict[str, object]]:
    query_rows = tuple(
        {
            "query_id": query.query_id,
            "relevant_doc_ids": list(query.relevant_doc_ids),
        }
        for query in cache.queries
    )
    task = {
        "primary_metric": "ndcg",
        "k": 10,
        "documents": [{}] * document_count,
        "queries": query_rows,
    }
    return task, {"query_rows": query_rows}


def _policy_row_for_scores(
    *,
    policy_name: str,
    weights_by_variant_name: dict[str, float],
    expansion_variant_name: str | None,
    expansion_weight: float | None,
    score_batch: tuple,
    task: dict[str, object],
    k: int,
) -> PolicyExpansionRow:
    ranked_doc_ids_by_query = tuple(
        _topk_doc_ids(scores.doc_ids, scores.values, k)
        for scores in score_batch
    )
    metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    return PolicyExpansionRow(
        policy_name=policy_name,
        weights_by_variant_name=dict(weights_by_variant_name),
        expansion_variant_name=expansion_variant_name,
        expansion_weight=expansion_weight,
        mean_ndcg_at_k=metrics.mean_ndcg_at_k,
        mean_recall_at_k=metrics.mean_recall_at_k,
        mean_reciprocal_rank=metrics.mean_reciprocal_rank,
        success_rate_at_k=metrics.success_rate_at_k,
    )


def _rescored_policy_row(
    *,
    policy_name: str,
    weights_by_variant_name: dict[str, float],
    expansion_variant_name: str,
    expansion_weight: float,
    encoded_queries_by_variant_name: dict[str, tuple],
    fused_full_scores: tuple,
    index: "LateIndex",
    task: dict[str, object],
    shortlist_k: int,
    match_k: int,
    k: int,
) -> PolicyExpansionRow:
    shortlist_doc_ids = tuple(
        _topk_doc_ids(scores.doc_ids, scores.values, shortlist_k)
        for scores in fused_full_scores
    )
    materialize_start = time.perf_counter()
    candidate_indices = tuple(
        index.select(doc_ids) for doc_ids in shortlist_doc_ids
    )
    shortlist_materialization_seconds = time.perf_counter() - materialize_start

    rescoring_start = time.perf_counter()
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for query_index, candidate_index in enumerate(candidate_indices):
        rescored_parts = []
        for variant_name, weight in weights_by_variant_name.items():
            rescored = topk_mean_similarity_scores(
                encoded_queries_by_variant_name[variant_name][query_index],
                candidate_index,
                match_k=match_k,
            )
            rescored_parts.append((weight, (rescored,)))
        fused_rescored = weighted_sum_score_batches(tuple(rescored_parts))[0]
        ranked_doc_ids_by_query.append(
            _topk_doc_ids(fused_rescored.doc_ids, fused_rescored.values, k)
        )
    rescoring_seconds = time.perf_counter() - rescoring_start

    metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    return PolicyExpansionRow(
        policy_name=policy_name,
        weights_by_variant_name=dict(weights_by_variant_name),
        expansion_variant_name=expansion_variant_name,
        expansion_weight=expansion_weight,
        mean_ndcg_at_k=metrics.mean_ndcg_at_k,
        mean_recall_at_k=metrics.mean_recall_at_k,
        mean_reciprocal_rank=metrics.mean_reciprocal_rank,
        success_rate_at_k=metrics.success_rate_at_k,
        shortlist_materialization_seconds=shortlist_materialization_seconds,
        rescoring_seconds=rescoring_seconds,
        rescoring_seconds_per_query=rescoring_seconds / len(shortlist_doc_ids),
    )


def _topk_doc_ids(
    doc_ids: tuple[str, ...],
    values: np.ndarray,
    k: int,
) -> tuple[str, ...]:
    if k < 0:
        raise ValueError("top-k requires a non-negative k")
    if k == 0:
        return ()

    effective_k = min(k, values.size)
    if effective_k == values.size:
        candidate_indices = np.arange(values.size, dtype=np.int64)
    else:
        candidate_indices = np.argpartition(
            values,
            values.size - effective_k,
        )[-effective_k:]

    # Preserve descending score order while breaking ties by original index.
    ordered = candidate_indices[
        np.lexsort((candidate_indices, -values[candidate_indices]))
    ]
    return tuple(doc_ids[int(index)] for index in ordered)


def benchmark_r2med_policy_expansions(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    variant_cache_root: Path = DEFAULT_VARIANT_CACHE_ROOT,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    seed_policy: LateInteractionPolicySpec | None = None,
    expansion_variant_names: tuple[str, ...] | None = None,
    expansion_weights: tuple[float, ...] | None = None,
    shortlist_k: int = 30,
    match_k: int = 4,
    k: int = 10,
    backend: str = kayak.MOJO_EXACT_CPU_BACKEND,
    force_rebuild_query_variants: bool = False,
) -> R2MEDPolicyExpansionSummary:
    if seed_policy is None:
        seed_policy = default_seed_policy()
    if expansion_variant_names is None:
        expansion_variant_names = default_expansion_variant_names()
    if expansion_weights is None:
        expansion_weights = default_expansion_weights()
    if shortlist_k < k:
        raise ValueError("shortlist_k must be at least k")

    store = DirectoryLateStore(snapshot_root)
    index = store.load_index(include_text=False)

    selected_variant_names = tuple(
        sorted(
            set(seed_policy.weights_by_variant_name).union(
                set(expansion_variant_names)
            )
        )
    )

    encoded_queries_by_variant_name: dict[str, tuple] = {}
    variant_score_seconds: dict[str, float] = {}
    full_scores_by_variant_name: dict[str, tuple] = {}
    reference_cache = None

    for variant_name in selected_variant_names:
        cache, _, _ = ensure_r2med_query_variant_cache(
            path=_variant_cache_path(variant_cache_root, variant_name),
            spec=default_r2med_query_variant(variant_name),
            dataset_id=dataset_id,
            model_name=model_name,
            force_rebuild=force_rebuild_query_variants,
        )
        if reference_cache is None:
            reference_cache = cache
        queries = tuple(
            kayak.query(query.vectors, text=query.text)
            for query in cache.queries
        )
        encoded_queries_by_variant_name[variant_name] = queries
        clear_prepared_packed_index_cache()
        start = time.perf_counter()
        full_scores_by_variant_name[variant_name] = kayak.maxsim_batch(
            kayak.query_batch(queries),
            index,
            backend=backend,
        )
        variant_score_seconds[variant_name] = time.perf_counter() - start
        clear_prepared_packed_index_cache()

    assert reference_cache is not None
    task, _ = _task_rows(reference_cache, document_count=index.document_count)

    single_variant_rows = tuple(
        sorted(
            (
                _policy_row_for_scores(
                    policy_name=variant_name,
                    weights_by_variant_name={variant_name: 1.0},
                    expansion_variant_name=None,
                    expansion_weight=None,
                    score_batch=full_scores_by_variant_name[variant_name],
                    task=task,
                    k=k,
                )
                for variant_name in selected_variant_names
            ),
            key=lambda row: row.mean_ndcg_at_k,
            reverse=True,
        )
    )

    maxsim_expansion_rows: list[PolicyExpansionRow] = []
    for expansion_variant_name in expansion_variant_names:
        for weight in expansion_weights:
            weights_by_variant_name = dict(seed_policy.weights_by_variant_name)
            weights_by_variant_name[expansion_variant_name] = weight
            policy_name = (
                f"{seed_policy.name}_plus_{weight:g}_{expansion_variant_name}"
            )
            fused_full_scores = weighted_sum_score_batches(
                tuple(
                    (
                        variant_weight,
                        full_scores_by_variant_name[variant_name],
                    )
                    for variant_name, variant_weight in (
                        weights_by_variant_name.items()
                    )
                )
            )
            maxsim_expansion_rows.append(
                _policy_row_for_scores(
                    policy_name=policy_name,
                    weights_by_variant_name=weights_by_variant_name,
                    expansion_variant_name=expansion_variant_name,
                    expansion_weight=weight,
                    score_batch=fused_full_scores,
                    task=task,
                    k=k,
                )
            )

    sorted_maxsim_expansion_rows = tuple(
        sorted(
            maxsim_expansion_rows,
            key=lambda row: row.mean_ndcg_at_k,
            reverse=True,
        )
    )
    best_maxsim_row = sorted_maxsim_expansion_rows[0]
    assert best_maxsim_row.expansion_variant_name is not None
    rescored_variant_name = best_maxsim_row.expansion_variant_name

    rescored_expansion_rows: list[PolicyExpansionRow] = []
    for weight in expansion_weights:
        weights_by_variant_name = dict(seed_policy.weights_by_variant_name)
        weights_by_variant_name[rescored_variant_name] = weight
        policy_name = f"{seed_policy.name}_plus_{weight:g}_{rescored_variant_name}"
        fused_full_scores = weighted_sum_score_batches(
            tuple(
                (
                    variant_weight,
                    full_scores_by_variant_name[variant_name],
                )
                for variant_name, variant_weight in (
                    weights_by_variant_name.items()
                )
            )
        )
        rescored_expansion_rows.append(
            _rescored_policy_row(
                policy_name=policy_name,
                weights_by_variant_name=weights_by_variant_name,
                expansion_variant_name=rescored_variant_name,
                expansion_weight=weight,
                encoded_queries_by_variant_name=encoded_queries_by_variant_name,
                fused_full_scores=fused_full_scores,
                index=index,
                task=task,
                shortlist_k=shortlist_k,
                match_k=match_k,
                k=k,
            )
        )

    return R2MEDPolicyExpansionSummary(
        dataset_id=dataset_id,
        model_name=model_name,
        document_count=index.document_count,
        query_count=len(task["queries"]),
        vector_dim=index.vector_dim,
        k=k,
        backend=backend,
        shortlist_k=shortlist_k,
        match_k=match_k,
        seed_policy_name=seed_policy.name,
        seed_weights_by_variant_name=dict(seed_policy.weights_by_variant_name),
        rescored_variant_name=rescored_variant_name,
        variant_score_seconds=dict(variant_score_seconds),
        single_variant_rows=single_variant_rows,
        maxsim_expansion_rows=sorted_maxsim_expansion_rows,
        rescored_expansion_rows=tuple(
            sorted(
                rescored_expansion_rows,
                key=lambda row: row.mean_ndcg_at_k,
                reverse=True,
            )
        ),
    )
