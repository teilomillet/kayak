from std.collections import List

from kayak.numeric import MetricScalar
from kayak.search import SearchHit


struct QueryEvaluation(Copyable):
    var query_id: String
    var description: String
    var primary_metric: String
    var primary_value: MetricScalar
    var k: Int
    var relevant_doc_count: Int
    var hit_count: Int
    var ndcg_at_k: MetricScalar
    var reciprocal_rank_at_k: MetricScalar
    var recall_at_k: MetricScalar
    var success_at_k: MetricScalar
    var hits: List[SearchHit]

    def __init__(
        out self,
        var query_id: String,
        var description: String,
        var primary_metric: String,
        primary_value: MetricScalar,
        k: Int,
        relevant_doc_count: Int,
        hit_count: Int,
        ndcg_at_k: MetricScalar,
        reciprocal_rank_at_k: MetricScalar,
        recall_at_k: MetricScalar,
        success_at_k: MetricScalar,
        var hits: List[SearchHit],
    ):
        self.query_id = query_id^
        self.description = description^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.k = k
        self.relevant_doc_count = relevant_doc_count
        self.hit_count = hit_count
        self.ndcg_at_k = ndcg_at_k
        self.reciprocal_rank_at_k = reciprocal_rank_at_k
        self.recall_at_k = recall_at_k
        self.success_at_k = success_at_k
        self.hits = hits^
