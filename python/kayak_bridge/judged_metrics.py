"""Owns judged retrieval metrics for ranked document ids.

This module keeps the benchmark-side metric semantics explicit and aligned with
Kayak's Mojo evaluation path. It does not own scoring or external-engine
execution; it only turns ranked doc ids plus judged relevance into stable
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Mapping, Sequence


def _discounted_gain_at_rank(index: int) -> float:
    return 1.0 / log2(index + 2)


def reciprocal_rank_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int
) -> float:
    for index, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_doc_ids:
            return 1.0 / float(index + 1)
    return 0.0


def success_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int
) -> float:
    for doc_id in ranked_doc_ids[:k]:
        if doc_id in relevant_doc_ids:
            return 1.0
    return 0.0


def recall_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int
) -> float:
    if not relevant_doc_ids:
        return 0.0

    found_count = 0
    for doc_id in ranked_doc_ids[:k]:
        if doc_id in relevant_doc_ids:
            found_count += 1

    return float(found_count) / float(len(relevant_doc_ids))


def ndcg_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int
) -> float:
    if not relevant_doc_ids:
        return 0.0

    dcg = 0.0
    for index, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_doc_ids:
            dcg += _discounted_gain_at_rank(index)

    ideal_limit = min(k, len(relevant_doc_ids))
    ideal_dcg = 0.0
    for index in range(ideal_limit):
        ideal_dcg += _discounted_gain_at_rank(index)

    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def choose_primary_value(
    primary_metric: str,
    *,
    mean_ndcg_at_k: float,
    mean_reciprocal_rank: float,
    mean_recall_at_k: float,
    success_rate_at_k: float,
) -> float:
    if primary_metric == "ndcg":
        return mean_ndcg_at_k
    if primary_metric == "mrr":
        return mean_reciprocal_rank
    if primary_metric == "recall":
        return mean_recall_at_k
    if primary_metric == "success":
        return success_rate_at_k
    raise ValueError(f"unknown primary metric: {primary_metric}")


@dataclass(frozen=True, slots=True)
class QueryMetricSummary:
    ndcg_at_k: float
    reciprocal_rank_at_k: float
    recall_at_k: float
    success_at_k: float


@dataclass(frozen=True, slots=True)
class TaskMetricSummary:
    primary_metric: str
    primary_value: float
    mean_ndcg_at_k: float
    mean_reciprocal_rank: float
    mean_recall_at_k: float
    success_rate_at_k: float
    k: int
    query_count: int
    document_count: int


def summarize_ranked_query(
    *,
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> QueryMetricSummary:
    return QueryMetricSummary(
        ndcg_at_k=ndcg_at_k(ranked_doc_ids, relevant_doc_ids, k),
        reciprocal_rank_at_k=reciprocal_rank_at_k(
            ranked_doc_ids, relevant_doc_ids, k
        ),
        recall_at_k=recall_at_k(ranked_doc_ids, relevant_doc_ids, k),
        success_at_k=success_at_k(ranked_doc_ids, relevant_doc_ids, k),
    )


def summarize_ranked_task(
    *,
    task: Mapping[str, object],
    ranked_doc_ids_by_query: Sequence[Sequence[str]],
) -> TaskMetricSummary:
    queries = task["queries"]
    if not isinstance(queries, Sequence):
        raise ValueError("task queries must be a sequence")
    if len(queries) == 0:
        raise ValueError("cannot evaluate a task with zero queries")
    if len(queries) != len(ranked_doc_ids_by_query):
        raise ValueError("query rankings must align with task queries")

    k = int(task["k"])
    ndcg_total = 0.0
    reciprocal_rank_total = 0.0
    recall_total = 0.0
    success_total = 0.0

    for query, ranked_doc_ids in zip(queries, ranked_doc_ids_by_query, strict=True):
        if not isinstance(query, Mapping):
            raise ValueError("task query entries must be mappings")
        summary = summarize_ranked_query(
            ranked_doc_ids=ranked_doc_ids,
            relevant_doc_ids=query["relevant_doc_ids"],
            k=k,
        )
        ndcg_total += summary.ndcg_at_k
        reciprocal_rank_total += summary.reciprocal_rank_at_k
        recall_total += summary.recall_at_k
        success_total += summary.success_at_k

    query_count = len(queries)
    mean_ndcg_at_k = ndcg_total / float(query_count)
    mean_reciprocal_rank = reciprocal_rank_total / float(query_count)
    mean_recall_at_k = recall_total / float(query_count)
    success_rate_at_k = success_total / float(query_count)
    primary_metric = str(task["primary_metric"])

    return TaskMetricSummary(
        primary_metric=primary_metric,
        primary_value=choose_primary_value(
            primary_metric,
            mean_ndcg_at_k=mean_ndcg_at_k,
            mean_reciprocal_rank=mean_reciprocal_rank,
            mean_recall_at_k=mean_recall_at_k,
            success_rate_at_k=success_rate_at_k,
        ),
        mean_ndcg_at_k=mean_ndcg_at_k,
        mean_reciprocal_rank=mean_reciprocal_rank,
        mean_recall_at_k=mean_recall_at_k,
        success_rate_at_k=success_rate_at_k,
        k=k,
        query_count=query_count,
        document_count=len(task["documents"]),
    )
