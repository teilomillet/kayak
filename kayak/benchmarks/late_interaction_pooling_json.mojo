from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.eval import TaskEvaluation, choose_primary_value, evaluate_query_hits
from kayak.numeric import MetricScalar, zero_metric_scalar
from kayak.runtime import ExactCpuBackend
from kayak.scoring import (
    LATE_INTERACTION_POOLING_KIND_MAXSIM,
    LATE_INTERACTION_POOLING_KIND_TOPK_MEAN,
    require_late_interaction_pooling_kind_supported,
    topk_mean_scores_for_index,
)
from kayak.search import SearchHit, search_exact
from kayak.search.topk import top_k_hits
from kayak.storage import StoredJudgedTask, StoredPackedIndex

from .json_common import json_escape


comptime LATE_INTERACTION_POOLING_SEARCH_MIN_SECONDS = 0.05
comptime LATE_INTERACTION_POOLING_SEARCH_MAX_SECONDS = 0.25
comptime LATE_INTERACTION_POOLING_SEARCH_MAX_ITERS = 200


struct LateInteractionPoolingSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var primary_metric: String
    var primary_value: Float64
    var pooling_kind: String
    var match_k: Int
    var query_count: Int
    var document_count: Int
    var full_vector_count: Int
    var vector_dim: Int
    var mean_reference_recall_at_k: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var mean_search_seconds: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var primary_metric: String,
        primary_value: Float64,
        var pooling_kind: String,
        match_k: Int,
        query_count: Int,
        document_count: Int,
        full_vector_count: Int,
        vector_dim: Int,
        mean_reference_recall_at_k: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        mean_search_seconds: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.pooling_kind = pooling_kind^
        self.match_k = match_k
        self.query_count = query_count
        self.document_count = document_count
        self.full_vector_count = full_vector_count
        self.vector_dim = vector_dim
        self.mean_reference_recall_at_k = mean_reference_recall_at_k
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.mean_search_seconds = mean_search_seconds


def late_interaction_pooling_hits_for_query(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read index: StoredPackedIndex,
    k: Int,
    pooling_kind: String,
    match_k: Int,
) raises -> List[SearchHit]:
    var effective_kind = require_late_interaction_pooling_kind_supported(
        pooling_kind
    )
    if effective_kind == LATE_INTERACTION_POOLING_KIND_MAXSIM:
        return search_exact(backend, query, index.index, k)
    if effective_kind == LATE_INTERACTION_POOLING_KIND_TOPK_MEAN:
        return top_k_hits(
            index.index.doc_ids,
            topk_mean_scores_for_index(query, index.index, match_k),
            k,
        )

    raise Error("unsupported late interaction pooling kind: " + effective_kind)


def reference_recall_at_k(
    read reference_hits: List[SearchHit],
    read candidate_hits: List[SearchHit],
    k: Int,
) -> Float64:
    var overlap = 0
    var limit = k
    if limit > len(reference_hits):
        limit = len(reference_hits)

    if limit == 0:
        return 0.0

    for reference_index in range(limit):
        var doc_id = reference_hits[reference_index].doc_id
        for candidate_index in range(len(candidate_hits)):
            if candidate_hits[candidate_index].doc_id == doc_id:
                overlap += 1
                break

    return Float64(overlap) / Float64(limit)


def evaluate_task_with_late_interaction_pooling(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
    pooling_kind: String,
    match_k: Int,
) raises -> TaskEvaluation:
    var task = stored_task.task.copy()
    if len(task.queries) == 0:
        raise Error("cannot evaluate a task with zero queries")

    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()

    for judged_query in task.queries:
        var hits = late_interaction_pooling_hits_for_query(
            backend,
            judged_query.query,
            stored_index,
            task.k,
            pooling_kind,
            match_k,
        )
        var query_evaluation = evaluate_query_hits(
            judged_query, hits, task.k, task.primary_metric
        )
        ndcg_total += query_evaluation.ndcg_at_k
        reciprocal_rank_total += query_evaluation.reciprocal_rank_at_k
        recall_total += query_evaluation.recall_at_k
        success_total += query_evaluation.success_at_k

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


def build_late_interaction_pooling_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
    pooling_kind: String,
    match_k: Int,
) raises -> LateInteractionPoolingSummary:
    var effective_kind = require_late_interaction_pooling_kind_supported(
        pooling_kind
    )
    var effective_match_k = match_k
    if effective_kind == LATE_INTERACTION_POOLING_KIND_MAXSIM:
        effective_match_k = 1

    var task = stored_task.task.copy()
    var evaluation = evaluate_task_with_late_interaction_pooling(
        backend,
        stored_task,
        stored_index,
        effective_kind,
        effective_match_k,
    )
    var reference_recall_total = 0.0
    var query_index = 0

    for judged_query in task.queries:
        var reference_hits = search_exact(
            backend, judged_query.query, stored_index.index, task.k
        )
        var candidate_hits = late_interaction_pooling_hits_for_query(
            backend,
            judged_query.query,
            stored_index,
            task.k,
            effective_kind,
            effective_match_k,
        )
        reference_recall_total += reference_recall_at_k(
            reference_hits,
            candidate_hits,
            task.k,
        )

    def search_once() capturing raises:
        bench_compiler.keep(
            late_interaction_pooling_hits_for_query(
                backend,
                task.queries[query_index].query,
                stored_index,
                task.k,
                effective_kind,
                effective_match_k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var search_report = run[search_once](
        num_warmup_iters=1,
        max_iters=LATE_INTERACTION_POOLING_SEARCH_MAX_ITERS,
        min_runtime_secs=LATE_INTERACTION_POOLING_SEARCH_MIN_SECONDS,
        max_runtime_secs=LATE_INTERACTION_POOLING_SEARCH_MAX_SECONDS,
    )

    return LateInteractionPoolingSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        effective_kind,
        effective_match_k,
        len(task.queries),
        len(task.documents),
        stored_index.index.total_vector_count,
        stored_index.index.vector_dim,
        reference_recall_total / Float64(len(task.queries)),
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        Float64(search_report.mean()),
    )


def append_late_interaction_pooling_summary_json(
    mut buffer: String,
    read summary: LateInteractionPoolingSummary,
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"pooling_kind\":\"" + json_escape(summary.pooling_kind) + "\","
    buffer += "\"match_k\":" + String(summary.match_k) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"full_vector_count\":" + String(summary.full_vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"mean_reference_recall_at_k\":"
    buffer += String(summary.mean_reference_recall_at_k) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds)
    buffer += "}"


def late_interaction_pooling_summary_json(
    read summary: LateInteractionPoolingSummary
) -> String:
    var buffer = String()
    append_late_interaction_pooling_summary_json(buffer, summary)
    return buffer^


def late_interaction_pooling_summaries_json(
    read summaries: List[LateInteractionPoolingSummary]
) -> String:
    var buffer = String()
    buffer += "["
    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_late_interaction_pooling_summary_json(buffer, summaries[index])
    buffer += "]"
    return buffer^
