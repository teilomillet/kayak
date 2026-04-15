"""Owns shortlist rescoring sweeps for pure late-interaction operators.

This module owns:
- full-index candidate generation with exact late-interaction fused scores
- shortlist materialization timing separate from rescoring timing
- pure late-interaction operator sweeps beyond hard MaxSim

This module does not own:
- non-late-interaction rerankers
- corpus snapshot building
- public leaderboard scraping
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import kayak
from kayak.stores import DirectoryLateStore

from .colbert_encoder import DEFAULT_MODEL_NAME
from .judged_metrics import summarize_ranked_task
from .prepared_index_cache import clear_prepared_packed_index_cache
from .r2med_biology_full import (
    DEFAULT_DATASET_ID,
    DEFAULT_SNAPSHOT_ROOT,
)
from .r2med_biology_late_interaction_quality import (
    DEFAULT_VARIANT_CACHE_ROOT,
    LateInteractionPolicySpec,
    default_policy_specs,
)
from .r2med_biology_query_variants import (
    default_r2med_query_variant,
    ensure_r2med_query_variant_cache,
)
from .reference_softmax_pooling import softmax_similarity_scores
from .reference_topk_pooling import topk_mean_similarity_scores
from .score_fusion import weighted_sum_score_batches


@dataclass(frozen=True, slots=True)
class LateInteractionRescoreOperatorSpec:
    name: str
    kind: str
    match_k: int | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "topk_mean":
            if self.match_k is None or self.match_k <= 0:
                raise ValueError("topk_mean rescoring requires a positive match_k")
            if self.temperature is not None:
                raise ValueError("topk_mean rescoring does not use temperature")
            return

        if self.kind == "softmax":
            if self.temperature is None or self.temperature <= 0.0:
                raise ValueError(
                    "softmax rescoring requires a positive temperature"
                )
            if self.match_k is not None:
                raise ValueError("softmax rescoring does not use match_k")
            return

        raise ValueError(f"unsupported rescoring operator: {self.kind}")


@dataclass(frozen=True, slots=True)
class LateInteractionRescoreRow:
    operator_name: str
    operator_kind: str
    match_k: int | None
    temperature: float | None
    mean_ndcg_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    success_rate_at_k: float
    rescoring_seconds: float
    rescoring_seconds_per_query: float

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class R2MEDLateInteractionRescoreSummary:
    dataset_id: str
    model_name: str
    document_count: int
    query_count: int
    vector_dim: int
    k: int
    backend: str
    candidate_policy_name: str
    candidate_weights_by_variant_name: dict[str, float]
    candidate_variant_score_seconds: dict[str, float]
    shortlist_k: int
    shortlist_materialization_seconds: float
    shortlist_materialization_seconds_per_query: float
    mean_shortlist_document_vector_count: float
    rows: tuple[LateInteractionRescoreRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "document_count": self.document_count,
            "query_count": self.query_count,
            "vector_dim": self.vector_dim,
            "k": self.k,
            "backend": self.backend,
            "candidate_policy_name": self.candidate_policy_name,
            "candidate_weights_by_variant_name": dict(
                self.candidate_weights_by_variant_name
            ),
            "candidate_variant_score_seconds": dict(
                self.candidate_variant_score_seconds
            ),
            "shortlist_k": self.shortlist_k,
            "shortlist_materialization_seconds": self.shortlist_materialization_seconds,
            "shortlist_materialization_seconds_per_query": (
                self.shortlist_materialization_seconds_per_query
            ),
            "mean_shortlist_document_vector_count": (
                self.mean_shortlist_document_vector_count
            ),
            "rows": [row.to_json_ready() for row in self.rows],
        }


def default_candidate_policy() -> LateInteractionPolicySpec:
    for policy in default_policy_specs():
        if (
            policy.name
            == "query2doc_plus_1.9_lamer_plus_0.35_search_r1_qwen7b_ins"
        ):
            return policy
    raise RuntimeError(
        "default candidate policy "
        "query2doc_plus_1.9_lamer_plus_0.35_search_r1_qwen7b_ins missing"
    )


def default_rescore_operator_specs(
) -> tuple[LateInteractionRescoreOperatorSpec, ...]:
    return (
        LateInteractionRescoreOperatorSpec(
            name="maxsim_shortlist",
            kind="topk_mean",
            match_k=1,
        ),
        LateInteractionRescoreOperatorSpec(
            name="topk_mean_k2",
            kind="topk_mean",
            match_k=2,
        ),
        LateInteractionRescoreOperatorSpec(
            name="topk_mean_k4",
            kind="topk_mean",
            match_k=4,
        ),
        LateInteractionRescoreOperatorSpec(
            name="topk_mean_k8",
            kind="topk_mean",
            match_k=8,
        ),
        LateInteractionRescoreOperatorSpec(
            name="softmax_tau0.25",
            kind="softmax",
            temperature=0.25,
        ),
        LateInteractionRescoreOperatorSpec(
            name="softmax_tau0.5",
            kind="softmax",
            temperature=0.5,
        ),
        LateInteractionRescoreOperatorSpec(
            name="softmax_tau1.0",
            kind="softmax",
            temperature=1.0,
        ),
    )


def _variant_cache_path(cache_root: Path, variant_name: str) -> Path:
    return cache_root / f"{variant_name}.json"


def _score_candidate_index(
    query: "LateQuery",
    candidate_index: "LateIndex",
    *,
    spec: LateInteractionRescoreOperatorSpec,
) -> "LateScores":
    if spec.kind == "topk_mean":
        assert spec.match_k is not None
        return topk_mean_similarity_scores(
            query,
            candidate_index,
            match_k=spec.match_k,
        )

    if spec.kind == "softmax":
        assert spec.temperature is not None
        return softmax_similarity_scores(
            query,
            candidate_index,
            temperature=spec.temperature,
        )

    raise ValueError(f"unsupported rescoring operator: {spec.kind}")


def _mean_shortlist_document_vector_count(
    candidate_indices: tuple["LateIndex", ...],
) -> float:
    if not candidate_indices:
        return 0.0
    total = 0
    for index in candidate_indices:
        total += index.total_vector_count
    return total / len(candidate_indices)


def benchmark_r2med_late_interaction_rescoring(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    variant_cache_root: Path = DEFAULT_VARIANT_CACHE_ROOT,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    candidate_policy: LateInteractionPolicySpec | None = None,
    operator_specs: tuple[LateInteractionRescoreOperatorSpec, ...] | None = None,
    shortlist_k: int = 100,
    k: int = 10,
    backend: str = kayak.MOJO_EXACT_CPU_BACKEND,
    force_rebuild_query_variants: bool = False,
) -> R2MEDLateInteractionRescoreSummary:
    if shortlist_k < k:
        raise ValueError("shortlist_k must be at least k")
    if candidate_policy is None:
        candidate_policy = default_candidate_policy()
    if operator_specs is None:
        operator_specs = default_rescore_operator_specs()

    store = DirectoryLateStore(snapshot_root)
    index = store.load_index(include_text=False)

    variant_names = tuple(sorted(candidate_policy.weights_by_variant_name))
    encoded_variants: dict[str, object] = {}
    full_scores_by_variant_name: dict[str, tuple] = {}
    candidate_variant_score_seconds: dict[str, float] = {}

    for variant_name in variant_names:
        cache, _, _ = ensure_r2med_query_variant_cache(
            path=_variant_cache_path(variant_cache_root, variant_name),
            spec=default_r2med_query_variant(variant_name),
            dataset_id=dataset_id,
            model_name=model_name,
            force_rebuild=force_rebuild_query_variants,
        )
        encoded_variants[variant_name] = cache
        queries = tuple(
            kayak.query(query.vectors, text=query.text)
            for query in cache.queries
        )
        clear_prepared_packed_index_cache()
        start = time.perf_counter()
        full_scores_by_variant_name[variant_name] = kayak.maxsim_batch(
            kayak.query_batch(queries),
            index,
            backend=backend,
        )
        candidate_variant_score_seconds[variant_name] = (
            time.perf_counter() - start
        )
        clear_prepared_packed_index_cache()

    first_variant_name = variant_names[0]
    reference_queries = encoded_variants[first_variant_name].queries
    query_rows = [
        {
            "query_id": query.query_id,
            "relevant_doc_ids": list(query.relevant_doc_ids),
        }
        for query in reference_queries
    ]

    fused_full_scores = weighted_sum_score_batches(
        tuple(
            (
                weight,
                full_scores_by_variant_name[variant_name],
            )
            for variant_name, weight in candidate_policy.weights_by_variant_name.items()
        )
    )
    shortlist_doc_ids_by_query = tuple(
        tuple(hit.doc_id for hit in scores.topk(shortlist_k))
        for scores in fused_full_scores
    )

    start = time.perf_counter()
    candidate_indices = tuple(
        index.select(candidate_doc_ids)
        for candidate_doc_ids in shortlist_doc_ids_by_query
    )
    shortlist_materialization_seconds = time.perf_counter() - start

    rows: list[LateInteractionRescoreRow] = []
    for operator_spec in operator_specs:
        start = time.perf_counter()
        ranked_doc_ids_by_query: list[tuple[str, ...]] = []
        for query_index, candidate_index in enumerate(candidate_indices):
            weighted_score_batches = []
            for variant_name, weight in (
                candidate_policy.weights_by_variant_name.items()
            ):
                rescored = _score_candidate_index(
                    kayak.query(
                        encoded_variants[variant_name].queries[
                            query_index
                        ].vectors,
                        text=encoded_variants[variant_name].queries[
                            query_index
                        ].text,
                    ),
                    candidate_index,
                    spec=operator_spec,
                )
                weighted_score_batches.append((weight, (rescored,)))
            fused_scores = weighted_sum_score_batches(weighted_score_batches)[0]
            ranked_doc_ids_by_query.append(
                tuple(hit.doc_id for hit in fused_scores.topk(k))
            )
        rescoring_seconds = time.perf_counter() - start
        metrics = summarize_ranked_task(
            task={
                "primary_metric": "ndcg",
                "k": k,
                "documents": [{}] * index.document_count,
                "queries": query_rows,
            },
            ranked_doc_ids_by_query=ranked_doc_ids_by_query,
        )
        rows.append(
            LateInteractionRescoreRow(
                operator_name=operator_spec.name,
                operator_kind=operator_spec.kind,
                match_k=operator_spec.match_k,
                temperature=operator_spec.temperature,
                mean_ndcg_at_k=metrics.mean_ndcg_at_k,
                mean_recall_at_k=metrics.mean_recall_at_k,
                mean_reciprocal_rank=metrics.mean_reciprocal_rank,
                success_rate_at_k=metrics.success_rate_at_k,
                rescoring_seconds=rescoring_seconds,
                rescoring_seconds_per_query=rescoring_seconds / len(query_rows),
            )
        )

    return R2MEDLateInteractionRescoreSummary(
        dataset_id=dataset_id,
        model_name=model_name,
        document_count=index.document_count,
        query_count=len(query_rows),
        vector_dim=index.vector_dim,
        k=k,
        backend=backend,
        candidate_policy_name=candidate_policy.name,
        candidate_weights_by_variant_name=dict(
            candidate_policy.weights_by_variant_name
        ),
        candidate_variant_score_seconds=dict(candidate_variant_score_seconds),
        shortlist_k=shortlist_k,
        shortlist_materialization_seconds=shortlist_materialization_seconds,
        shortlist_materialization_seconds_per_query=(
            shortlist_materialization_seconds / len(query_rows)
        ),
        mean_shortlist_document_vector_count=(
            _mean_shortlist_document_vector_count(candidate_indices)
        ),
        rows=tuple(rows),
    )
