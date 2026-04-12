from std.collections import List

from kayak.numeric import MetricScalar
from kayak.search import SearchHit

from .judged_query import JudgedQuery
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k, success_at_k
from .primary_metric import choose_primary_value
from .query_evaluation import QueryEvaluation


def evaluate_query_hits(
    read judged_query: JudgedQuery,
    hits: List[SearchHit],
    k: Int,
    primary_metric: String = "ndcg",
) raises -> QueryEvaluation:
    var query_ndcg_at_k = ndcg_at_k(hits, judged_query.relevant_doc_ids, k)
    var query_reciprocal_rank_at_k = reciprocal_rank_at_k(
        hits, judged_query.relevant_doc_ids, k
    )
    var query_recall_at_k = recall_at_k(hits, judged_query.relevant_doc_ids, k)
    var query_success_at_k = success_at_k(hits, judged_query.relevant_doc_ids, k)

    return QueryEvaluation(
        judged_query.query_id.copy(),
        judged_query.description.copy(),
        primary_metric.copy(),
        choose_primary_value(
            primary_metric,
            query_ndcg_at_k,
            query_reciprocal_rank_at_k,
            query_recall_at_k,
            query_success_at_k,
        ),
        k,
        len(judged_query.relevant_doc_ids),
        len(hits),
        query_ndcg_at_k,
        query_reciprocal_rank_at_k,
        query_recall_at_k,
        query_success_at_k,
        hits.copy(),
    )
