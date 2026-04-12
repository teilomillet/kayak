from kayak.index import pack_documents
from kayak.numeric import MetricScalar, zero_metric_scalar
from kayak.runtime import ExactScoringBackend
from kayak.search import search_exact
from kayak.verifier import VerifierReranker, search_exact_with_verifier

from .judged_task import JudgedTask
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k, success_at_k
from .primary_metric import choose_primary_value


struct TaskEvaluation(Copyable):
    var family: String
    var slice_name: String
    var primary_metric: String
    var primary_value: MetricScalar
    var k: Int
    var query_count: Int
    var document_count: Int
    var mean_ndcg_at_k: MetricScalar
    var mean_reciprocal_rank: MetricScalar
    var mean_recall_at_k: MetricScalar
    var success_rate_at_k: MetricScalar

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var primary_metric: String,
        primary_value: MetricScalar,
        k: Int,
        query_count: Int,
        document_count: Int,
        mean_ndcg_at_k: MetricScalar,
        mean_reciprocal_rank: MetricScalar,
        mean_recall_at_k: MetricScalar,
        success_rate_at_k: MetricScalar,
    ):
        self.family = family^
        self.slice_name = slice_name^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.k = k
        self.query_count = query_count
        self.document_count = document_count
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k


def evaluate_task[Backend: ExactScoringBackend](
    read backend: Backend, read task: JudgedTask
) raises -> TaskEvaluation:
    if len(task.queries) == 0:
        raise Error("cannot evaluate a task with zero queries")

    var index = pack_documents(task.documents)
    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()

    for judged_query in task.queries:
        var hits = search_exact(backend, judged_query.query, index, task.k)

        ndcg_total += ndcg_at_k(hits, judged_query.relevant_doc_ids, task.k)
        reciprocal_rank_total += reciprocal_rank_at_k(
            hits, judged_query.relevant_doc_ids, task.k
        )
        recall_total += recall_at_k(hits, judged_query.relevant_doc_ids, task.k)
        success_total += success_at_k(
            hits, judged_query.relevant_doc_ids, task.k
        )

    var query_count = len(task.queries)
    var mean_ndcg_at_k = ndcg_total / MetricScalar(query_count)
    var mean_reciprocal_rank = reciprocal_rank_total / MetricScalar(query_count)
    var mean_recall_at_k = recall_total / MetricScalar(query_count)
    var success_rate_at_k = success_total / MetricScalar(query_count)

    return TaskEvaluation(
        task.family.copy(),
        task.slice_name.copy(),
        task.primary_metric.copy(),
        choose_primary_value(
            task.primary_metric,
            mean_ndcg_at_k,
            mean_reciprocal_rank,
            mean_recall_at_k,
            success_rate_at_k,
        ),
        task.k,
        query_count,
        len(task.documents),
        mean_ndcg_at_k,
        mean_reciprocal_rank,
        mean_recall_at_k,
        success_rate_at_k,
    )


def evaluate_task_with_verifier[Backend: ExactScoringBackend](
    read backend: Backend,
    read task: JudgedTask,
    read verifier: VerifierReranker,
) raises -> TaskEvaluation:
    if len(task.queries) == 0:
        raise Error("cannot evaluate a task with zero queries")

    var index = pack_documents(task.documents)
    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()

    for judged_query in task.queries:
        var hits = search_exact_with_verifier(
            backend, judged_query.query, index, task.k, verifier
        )

        ndcg_total += ndcg_at_k(hits, judged_query.relevant_doc_ids, task.k)
        reciprocal_rank_total += reciprocal_rank_at_k(
            hits, judged_query.relevant_doc_ids, task.k
        )
        recall_total += recall_at_k(hits, judged_query.relevant_doc_ids, task.k)
        success_total += success_at_k(
            hits, judged_query.relevant_doc_ids, task.k
        )

    var query_count = len(task.queries)
    var mean_ndcg_at_k = ndcg_total / MetricScalar(query_count)
    var mean_reciprocal_rank = reciprocal_rank_total / MetricScalar(query_count)
    var mean_recall_at_k = recall_total / MetricScalar(query_count)
    var success_rate_at_k = success_total / MetricScalar(query_count)

    return TaskEvaluation(
        task.family.copy(),
        task.slice_name.copy(),
        task.primary_metric.copy(),
        choose_primary_value(
            task.primary_metric,
            mean_ndcg_at_k,
            mean_reciprocal_rank,
            mean_recall_at_k,
            success_rate_at_k,
        ),
        task.k,
        query_count,
        len(task.documents),
        mean_ndcg_at_k,
        mean_reciprocal_rank,
        mean_recall_at_k,
        success_rate_at_k,
    )
