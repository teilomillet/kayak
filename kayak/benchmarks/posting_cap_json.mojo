from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.eval import JudgedTask, evaluate_query_hits
from kayak.planning import (
    SearchPlan,
    candidate_generation_for_plan,
    explain_collection_search,
    final_hits_to_search_hits,
)
from kayak.runtime import ExactCpuBackend

from .json_common import json_escape
from .query_text_support import judged_query_text_for_plan
from .vector_budget_json import truncate_query_to_vector_budget


struct PostingCapSweepSummary(Copyable):
    var dataset_id: String
    var collection_id: String
    var snapshot_id: String
    var model_name: String
    var candidate_generator_kind: String
    var final_k: Int
    var candidate_k: Int
    var query_vector_budget: Int
    var document_vector_budget: Int
    var posting_cap: Int
    var query_count: Int
    var mean_candidate_generation_seconds: Float64
    var mean_candidate_recall_at_final_k: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var stage1_vector_count: Int
    var stage1_token_count: Int
    var stage1_byte_size: Int

    def __init__(
        out self,
        var dataset_id: String,
        var collection_id: String,
        var snapshot_id: String,
        var model_name: String,
        var candidate_generator_kind: String,
        final_k: Int,
        candidate_k: Int,
        query_vector_budget: Int,
        document_vector_budget: Int,
        posting_cap: Int,
        query_count: Int,
        mean_candidate_generation_seconds: Float64,
        mean_candidate_recall_at_final_k: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        stage1_vector_count: Int,
        stage1_token_count: Int,
        stage1_byte_size: Int,
    ):
        self.dataset_id = dataset_id^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.model_name = model_name^
        self.candidate_generator_kind = candidate_generator_kind^
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_vector_budget = query_vector_budget
        self.document_vector_budget = document_vector_budget
        self.posting_cap = posting_cap
        self.query_count = query_count
        self.mean_candidate_generation_seconds = mean_candidate_generation_seconds
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.stage1_vector_count = stage1_vector_count
        self.stage1_token_count = stage1_token_count
        self.stage1_byte_size = stage1_byte_size


def standard_posting_cap_sizes(max_posting_cap: Int) -> List[Int]:
    var sizes = List[Int]()

    for candidate_size in [1, 2, 4, 8, 16, 32, max_posting_cap]:
        var size = candidate_size
        if size > max_posting_cap:
            size = max_posting_cap
        if size <= 0:
            continue
        if len(sizes) == 0 or sizes[len(sizes) - 1] != size:
            sizes.append(size)

    return sizes^


def build_posting_cap_sweep_summary_for_plan(
    read backend: ExactCpuBackend,
    dataset_id: String,
    model_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    query_vector_budget: Int,
    document_vector_budget: Int,
    posting_cap: Int,
) raises -> PostingCapSweepSummary:
    var candidate_recall_total = 0.0
    var ndcg_total = 0.0
    var reciprocal_rank_total = 0.0
    var recall_total = 0.0
    var success_total = 0.0
    var stage1_vector_count = 0
    var stage1_token_count = 0
    var stage1_byte_size = 0
    var query_index = 0

    def candidate_once() capturing raises:
        bench_compiler.keep(
            candidate_generation_for_plan(
                backend,
                truncate_query_to_vector_budget(
                    task.queries[query_index].query, query_vector_budget
                ),
                snapshot,
                plan,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var candidate_report = run[candidate_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )

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

        candidate_recall_total += Float64(explain.candidate_recall_at_final_k)
        ndcg_total += Float64(query_evaluation.ndcg_at_k)
        reciprocal_rank_total += Float64(query_evaluation.reciprocal_rank_at_k)
        recall_total += Float64(query_evaluation.recall_at_k)
        success_total += Float64(query_evaluation.success_at_k)

        if stage1_vector_count == 0 and stage1_token_count == 0 and stage1_byte_size == 0:
            stage1_vector_count = explain.candidate_stage.vector_count
            stage1_token_count = explain.candidate_stage.token_count
            stage1_byte_size = explain.candidate_stage.byte_size

    var query_count = len(task.queries)
    return PostingCapSweepSummary(
        dataset_id,
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        model_name,
        plan.candidate_generator.kind.copy(),
        task.k,
        plan.candidate_budget.candidate_k,
        query_vector_budget,
        document_vector_budget,
        posting_cap,
        query_count,
        Float64(candidate_report.mean()),
        candidate_recall_total / Float64(query_count),
        ndcg_total / Float64(query_count),
        reciprocal_rank_total / Float64(query_count),
        recall_total / Float64(query_count),
        success_total / Float64(query_count),
        stage1_vector_count,
        stage1_token_count,
        stage1_byte_size,
    )


def append_posting_cap_sweep_summary_json(
    mut buffer: String,
    read summary: PostingCapSweepSummary,
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(summary.candidate_generator_kind) + "\","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_vector_budget\":" + String(summary.query_vector_budget) + ","
    buffer += "\"document_vector_budget\":"
    buffer += String(summary.document_vector_budget) + ","
    buffer += "\"posting_cap\":" + String(summary.posting_cap) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"mean_candidate_generation_seconds\":"
    buffer += String(summary.mean_candidate_generation_seconds) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"stage1_vector_count\":" + String(summary.stage1_vector_count) + ","
    buffer += "\"stage1_token_count\":" + String(summary.stage1_token_count) + ","
    buffer += "\"stage1_byte_size\":" + String(summary.stage1_byte_size)
    buffer += "}"


def posting_cap_sweep_summaries_json(
    read summaries: List[PostingCapSweepSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_posting_cap_sweep_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
