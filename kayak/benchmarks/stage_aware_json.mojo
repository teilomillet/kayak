from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.eval import evaluate_query_hits
from kayak.planning import (
    SearchPlan,
    explain_collection_search,
    final_hits_to_search_hits,
    search_collection_for_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import StoredJudgedTask

from .json_common import json_escape


struct StageAwareSearchSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var collection_id: String
    var snapshot_id: String
    var candidate_generator_kind: String
    var faithfulness_policy_kind: String
    var primary_metric: String
    var primary_value: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var mean_candidate_recall_at_final_k: Float64
    var mean_search_seconds: Float64
    var final_k: Int
    var candidate_k: Int
    var query_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int
    var bytes_per_document: Float64
    var bytes_per_vector: Float64
    var vector_dim: Int

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var collection_id: String,
        var snapshot_id: String,
        var candidate_generator_kind: String,
        var faithfulness_policy_kind: String,
        var primary_metric: String,
        primary_value: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        mean_candidate_recall_at_final_k: Float64,
        mean_search_seconds: Float64,
        final_k: Int,
        candidate_k: Int,
        query_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
        bytes_per_document: Float64,
        bytes_per_vector: Float64,
        vector_dim: Int,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.candidate_generator_kind = candidate_generator_kind^
        self.faithfulness_policy_kind = faithfulness_policy_kind^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k
        self.mean_search_seconds = mean_search_seconds
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_count = query_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
        self.bytes_per_document = bytes_per_document
        self.bytes_per_vector = bytes_per_vector
        self.vector_dim = vector_dim


def density_bytes_per_document(byte_size: Int, document_count: Int) -> Float64:
    if document_count == 0:
        return 0.0

    return Float64(byte_size) / Float64(document_count)


def density_bytes_per_vector(byte_size: Int, vector_count: Int) -> Float64:
    if vector_count == 0:
        return 0.0

    return Float64(byte_size) / Float64(vector_count)


def build_stage_aware_search_summary_from_measurement(
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    primary_value: Float64,
    mean_ndcg_at_k: Float64,
    mean_reciprocal_rank: Float64,
    mean_recall_at_k: Float64,
    success_rate_at_k: Float64,
    mean_candidate_recall_at_final_k: Float64,
    mean_search_seconds: Float64,
) -> StageAwareSearchSummary:
    var task = stored_task.task.copy()
    var stats = snapshot.snapshot.stats.copy()

    return StageAwareSearchSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        plan.candidate_generator.kind.copy(),
        plan.faithfulness_policy.kind.copy(),
        task.primary_metric.copy(),
        primary_value,
        mean_ndcg_at_k,
        mean_reciprocal_rank,
        mean_recall_at_k,
        success_rate_at_k,
        mean_candidate_recall_at_final_k,
        mean_search_seconds,
        plan.candidate_budget.final_k,
        plan.candidate_budget.candidate_k,
        len(task.queries),
        stats.document_count,
        stats.token_count,
        stats.total_vector_count,
        stats.byte_size,
        density_bytes_per_document(stats.byte_size, stats.document_count),
        density_bytes_per_vector(stats.byte_size, stats.total_vector_count),
        snapshot.collection.vector_dim,
    )


def build_stage_aware_search_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> StageAwareSearchSummary:
    var task = stored_task.task.copy()
    var primary_total = 0.0
    var ndcg_total = 0.0
    var reciprocal_rank_total = 0.0
    var recall_total = 0.0
    var success_total = 0.0
    var candidate_recall_total = 0.0

    for judged_query in task.queries:
        var final_hits = search_collection_for_plan(
            backend,
            judged_query.query,
            snapshot,
            plan,
        )
        var query_evaluation = evaluate_query_hits(
            judged_query,
            final_hits_to_search_hits(final_hits),
            task.k,
            task.primary_metric,
        )
        primary_total += Float64(query_evaluation.primary_value)
        ndcg_total += Float64(query_evaluation.ndcg_at_k)
        reciprocal_rank_total += Float64(
            query_evaluation.reciprocal_rank_at_k
        )
        recall_total += Float64(query_evaluation.recall_at_k)
        success_total += Float64(query_evaluation.success_at_k)

        var explain = explain_collection_search(
            backend,
            judged_query.query,
            snapshot,
            plan,
        )
        candidate_recall_total += Float64(explain.candidate_recall_at_final_k)

    var query_count = len(task.queries)
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(
            search_collection_for_plan(
                backend,
                task.queries[query_index].query,
                snapshot,
                plan,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = run[search_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    return build_stage_aware_search_summary_from_measurement(
        stored_task,
        snapshot,
        plan,
        primary_total / Float64(query_count),
        ndcg_total / Float64(query_count),
        reciprocal_rank_total / Float64(query_count),
        recall_total / Float64(query_count),
        success_total / Float64(query_count),
        candidate_recall_total / Float64(query_count),
        Float64(report.mean()),
    )


def append_stage_aware_search_summary_json(
    mut buffer: String,
    read summary: StageAwareSearchSummary,
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(summary.candidate_generator_kind) + "\","
    buffer += "\"faithfulness_policy_kind\":\""
    buffer += json_escape(summary.faithfulness_policy_kind) + "\","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":" + String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds) + ","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"token_count\":" + String(summary.token_count) + ","
    buffer += "\"vector_count\":" + String(summary.vector_count) + ","
    buffer += "\"byte_size\":" + String(summary.byte_size) + ","
    buffer += "\"bytes_per_document\":"
    buffer += String(summary.bytes_per_document) + ","
    buffer += "\"bytes_per_vector\":" + String(summary.bytes_per_vector) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim)
    buffer += "}"


def stage_aware_search_summary_json(read summary: StageAwareSearchSummary) -> String:
    var buffer = String()
    append_stage_aware_search_summary_json(buffer, summary)
    return buffer^


def stage_aware_search_summaries_json(
    read summaries: List[StageAwareSearchSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_stage_aware_search_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
