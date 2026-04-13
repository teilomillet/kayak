from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.eval import JudgedTask, evaluate_query_hits
from kayak.numeric import MetricScalar, VectorScalar, zero_metric_scalar
from kayak.planning import (
    CandidateGenerator,
    SearchPlan,
    best_effort_faithfulness_policy,
    document_proxy_search_plan,
    explain_collection_search,
    final_hits_to_search_hits,
)
from kayak.runtime import ExactCpuBackend

from .json_common import json_escape
from .query_text_support import judged_query_text_for_plan
from .search_plan_semantics_json import (
    append_candidate_generator_semantics_json_fields,
)


struct VectorBudgetSweepSummary(Copyable):
    var dataset_id: String
    var collection_id: String
    var snapshot_id: String
    var model_name: String
    var candidate_generator: CandidateGenerator
    var final_k: Int
    var candidate_k: Int
    var query_vector_budget: Int
    var document_vector_budget: Int
    var query_count: Int
    var mean_candidate_recall_at_final_k: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var collection_id: String,
        var snapshot_id: String,
        var model_name: String,
        candidate_generator: CandidateGenerator,
        final_k: Int,
        candidate_k: Int,
        query_vector_budget: Int,
        document_vector_budget: Int,
        query_count: Int,
        mean_candidate_recall_at_final_k: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
    ):
        self.dataset_id = dataset_id^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.model_name = model_name^
        self.candidate_generator = candidate_generator.copy()
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_vector_budget = query_vector_budget
        self.document_vector_budget = document_vector_budget
        self.query_count = query_count
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k


def capped_vector_budget(requested_budget: Int, available_count: Int) -> Int:
    if requested_budget == 0 or requested_budget > available_count:
        return available_count

    return requested_budget


def truncate_query_to_vector_budget(
    read query: EncodedQuery, requested_budget: Int
) raises -> EncodedQuery:
    var budget = capped_vector_budget(requested_budget, query.vector_count)
    var token_vectors = List[List[VectorScalar]]()

    for vector_index in range(budget):
        token_vectors.append(query.token_vectors[vector_index].copy())

    return EncodedQuery(token_vectors^)


def standard_query_vector_budget_sizes(nominal_query_vector_count: Int) -> List[Int]:
    var sizes = List[Int]()

    for candidate_size in [4, 8, 16, 32, nominal_query_vector_count]:
        var size = candidate_size
        if size > nominal_query_vector_count:
            size = nominal_query_vector_count
        if size <= 0:
            continue
        if len(sizes) == 0 or sizes[len(sizes) - 1] != size:
            sizes.append(size)

    return sizes^


def standard_document_vector_budget_sizes(
    nominal_document_vector_count: Int
) -> List[Int]:
    var sizes = List[Int]()

    for candidate_size in [8, 16, 32, 64, nominal_document_vector_count]:
        var size = candidate_size
        if size > nominal_document_vector_count:
            size = nominal_document_vector_count
        if size <= 0:
            continue
        if len(sizes) == 0 or sizes[len(sizes) - 1] != size:
            sizes.append(size)

    return sizes^


def build_vector_budget_sweep_summary(
    read backend: ExactCpuBackend,
    dataset_id: String,
    model_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    query_vector_budget: Int,
    document_vector_budget: Int,
    candidate_k: Int,
) raises -> VectorBudgetSweepSummary:
    return build_vector_budget_sweep_summary_for_plan(
        backend,
        dataset_id,
        model_name,
        task,
        snapshot,
        document_proxy_search_plan(
            task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        query_vector_budget,
        document_vector_budget,
    )


def build_vector_budget_sweep_summary_for_plan(
    read backend: ExactCpuBackend,
    dataset_id: String,
    model_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    query_vector_budget: Int,
    document_vector_budget: Int,
) raises -> VectorBudgetSweepSummary:
    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()
    var candidate_recall_total = 0.0

    for judged_query in task.queries:
        var budgeted_query = truncate_query_to_vector_budget(
            judged_query.query, query_vector_budget
        )
        var explain = explain_collection_search(
            backend,
            budgeted_query,
            snapshot,
            plan,
            query_text=judged_query_text_for_plan(
                plan,
                judged_query.description,
            ),
        )
        var query_evaluation = evaluate_query_hits(
            judged_query,
            final_hits_to_search_hits(explain.final_hits),
            task.k,
            task.primary_metric,
        )
        ndcg_total += query_evaluation.ndcg_at_k
        reciprocal_rank_total += query_evaluation.reciprocal_rank_at_k
        recall_total += query_evaluation.recall_at_k
        success_total += query_evaluation.success_at_k
        candidate_recall_total += Float64(explain.candidate_recall_at_final_k)

    var query_count = len(task.queries)

    return VectorBudgetSweepSummary(
        dataset_id,
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        model_name,
        plan.candidate_generator,
        task.k,
        plan.candidate_budget.candidate_k,
        query_vector_budget,
        document_vector_budget,
        query_count,
        candidate_recall_total / Float64(query_count),
        Float64(ndcg_total / MetricScalar(query_count)),
        Float64(reciprocal_rank_total / MetricScalar(query_count)),
        Float64(recall_total / MetricScalar(query_count)),
        Float64(success_total / MetricScalar(query_count)),
    )


def append_vector_budget_sweep_summary_json(
    mut buffer: String, read summary: VectorBudgetSweepSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    append_candidate_generator_semantics_json_fields(
        buffer,
        summary.candidate_generator,
    )
    buffer += ","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_vector_budget\":" + String(summary.query_vector_budget) + ","
    buffer += "\"document_vector_budget\":"
    buffer += String(summary.document_vector_budget) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k)
    buffer += "}"


def vector_budget_sweep_summaries_json(
    read summaries: List[VectorBudgetSweepSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_vector_budget_sweep_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
