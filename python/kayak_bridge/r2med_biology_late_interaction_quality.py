"""Owns pure late-interaction R2MED/Biology quality sweeps.

This module owns:
- single-variant and weighted-fusion evaluation on the full R2MED snapshot
- exact MaxSim-only policy benchmarking

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
from .r2med_biology_query_variants import (
    R2MEDQueryVariantSpec,
    default_r2med_query_variant,
    ensure_r2med_query_variant_cache,
)
from .score_fusion import weighted_sum_score_batches


DEFAULT_VARIANT_CACHE_ROOT = (
    Path(DEFAULT_SNAPSHOT_ROOT).parents[0] / "query_variants"
)


@dataclass(frozen=True, slots=True)
class LateInteractionPolicySpec:
    name: str
    weights_by_variant_name: dict[str, float]


@dataclass(frozen=True, slots=True)
class LateInteractionPolicyRow:
    policy_name: str
    weights_by_variant_name: dict[str, float]
    mean_ndcg_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    success_rate_at_k: float

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class R2MEDLateInteractionQualitySummary:
    dataset_id: str
    model_name: str
    document_count: int
    query_count: int
    vector_dim: int
    k: int
    backend: str
    variant_score_seconds: dict[str, float]
    rows: tuple[LateInteractionPolicyRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "document_count": self.document_count,
            "query_count": self.query_count,
            "vector_dim": self.vector_dim,
            "k": self.k,
            "backend": self.backend,
            "variant_score_seconds": dict(self.variant_score_seconds),
            "rows": [row.to_json_ready() for row in self.rows],
        }


def default_policy_specs() -> tuple[LateInteractionPolicySpec, ...]:
    return (
        LateInteractionPolicySpec("query", {"query": 1.0}),
        LateInteractionPolicySpec("hyde_gpt4", {"hyde_gpt4": 1.0}),
        LateInteractionPolicySpec("query2doc_gpt4", {"query2doc_gpt4": 1.0}),
        LateInteractionPolicySpec("lamer_gpt4", {"lamer_gpt4": 1.0}),
        LateInteractionPolicySpec(
            "query_plus_lamer",
            {"query": 1.0, "lamer_gpt4": 1.0},
        ),
        LateInteractionPolicySpec(
            "query2doc_plus_lamer",
            {"query2doc_gpt4": 1.0, "lamer_gpt4": 1.0},
        ),
        LateInteractionPolicySpec(
            "query2doc_plus_1.25_lamer",
            {"query2doc_gpt4": 1.0, "lamer_gpt4": 1.25},
        ),
        LateInteractionPolicySpec(
            "query2doc_plus_1.5_lamer",
            {"query2doc_gpt4": 1.0, "lamer_gpt4": 1.5},
        ),
        LateInteractionPolicySpec(
            "query2doc_plus_1.75_lamer",
            {"query2doc_gpt4": 1.0, "lamer_gpt4": 1.75},
        ),
        LateInteractionPolicySpec(
            "query2doc_plus_1.9_lamer",
            {"query2doc_gpt4": 1.0, "lamer_gpt4": 1.9},
        ),
        LateInteractionPolicySpec(
            "query2doc_plus_2.0_lamer",
            {"query2doc_gpt4": 1.0, "lamer_gpt4": 2.0},
        ),
    )


def _variant_cache_path(
    cache_root: Path,
    spec: R2MEDQueryVariantSpec,
) -> Path:
    return cache_root / f"{spec.name}.json"


def benchmark_r2med_late_interaction_policies(
    *,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
    variant_cache_root: Path = DEFAULT_VARIANT_CACHE_ROOT,
    dataset_id: str = DEFAULT_DATASET_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    policy_specs: tuple[LateInteractionPolicySpec, ...] | None = None,
    k: int = 10,
    backend: str = kayak.MOJO_EXACT_CPU_BACKEND,
    force_rebuild_query_variants: bool = False,
) -> R2MEDLateInteractionQualitySummary:
    if policy_specs is None:
        policy_specs = default_policy_specs()

    store = DirectoryLateStore(snapshot_root)
    index = store.load_index(include_text=False)

    variant_names = {
        variant_name
        for policy in policy_specs
        for variant_name in policy.weights_by_variant_name
    }

    encoded_variants: dict[str, object] = {}
    variant_score_seconds: dict[str, float] = {}
    scores_by_variant_name: dict[str, tuple] = {}

    for variant_name in sorted(variant_names):
        spec = default_r2med_query_variant(variant_name)
        cache, _, _ = ensure_r2med_query_variant_cache(
            path=_variant_cache_path(variant_cache_root, spec),
            spec=spec,
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
        scores_by_variant_name[variant_name] = kayak.maxsim_batch(
            kayak.query_batch(queries),
            index,
            backend=backend,
        )
        variant_score_seconds[variant_name] = time.perf_counter() - start
        clear_prepared_packed_index_cache()

    first_variant_name = next(iter(sorted(variant_names)))
    reference_queries = encoded_variants[first_variant_name].queries
    query_rows = [
        {
            "query_id": query.query_id,
            "relevant_doc_ids": list(query.relevant_doc_ids),
        }
        for query in reference_queries
    ]

    rows: list[LateInteractionPolicyRow] = []
    for policy in policy_specs:
        fused_scores = weighted_sum_score_batches(
            tuple(
                (
                    weight,
                    scores_by_variant_name[variant_name],
                )
                for variant_name, weight in policy.weights_by_variant_name.items()
            )
        )
        ranked_doc_ids_by_query = [
            tuple(hit.doc_id for hit in scores.topk(k))
            for scores in fused_scores
        ]
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
            LateInteractionPolicyRow(
                policy_name=policy.name,
                weights_by_variant_name=dict(policy.weights_by_variant_name),
                mean_ndcg_at_k=metrics.mean_ndcg_at_k,
                mean_recall_at_k=metrics.mean_recall_at_k,
                mean_reciprocal_rank=metrics.mean_reciprocal_rank,
                success_rate_at_k=metrics.success_rate_at_k,
            )
        )

    return R2MEDLateInteractionQualitySummary(
        dataset_id=dataset_id,
        model_name=model_name,
        document_count=index.document_count,
        query_count=len(query_rows),
        vector_dim=index.vector_dim,
        k=k,
        backend=backend,
        variant_score_seconds=dict(variant_score_seconds),
        rows=tuple(rows),
    )
