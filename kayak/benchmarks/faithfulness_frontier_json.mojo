# Stage-1 frontier benchmark output that keeps latency, recall, quality,
# and stage-1 storage in the same artifact.

from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.eval import evaluate_query_hits
from kayak.planning import (
    CandidateGenerator,
    SearchPlan,
    candidate_generation_for_plan,
    explain_collection_search,
    final_hits_to_search_hits,
    search_collection_for_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import StoredJudgedTask

from .json_common import json_escape
from .query_text_support import judged_query_text_for_plan
from .search_plan_semantics_json import (
    append_candidate_generator_semantics_json_fields,
)
from .vector_budget_json import truncate_query_to_vector_budget


struct FaithfulnessFrontierSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var collection_id: String
    var snapshot_id: String
    var candidate_generator: CandidateGenerator
    var final_k: Int
    var candidate_k: Int
    var query_vector_budget: Int
    var requested_stage1_vector_budget: Int
    var posting_cap: Int
    var query_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var vector_dim: Int
    var mean_candidate_generation_seconds: Float64
    var mean_search_seconds: Float64
    var mean_candidate_hit_count: Float64
    var mean_candidate_recall_at_final_k: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var stage1_vector_count: Int
    var stage1_token_count: Int
    var stage1_byte_size: Int
    var stage1_bytes_per_document: Float64
    var stage1_bytes_per_vector: Float64
    var mean_stage1_graph_visited_vertex_count: Float64
    var mean_stage1_graph_expanded_edge_count: Float64
    var mean_stage1_graph_visited_cluster_count: Float64
    var mean_stage1_graph_entry_point_count: Float64
    var mean_stage1_graph_max_frontier_size: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var collection_id: String,
        var snapshot_id: String,
        candidate_generator: CandidateGenerator,
        final_k: Int,
        candidate_k: Int,
        query_vector_budget: Int,
        requested_stage1_vector_budget: Int,
        posting_cap: Int,
        query_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        vector_dim: Int,
        mean_candidate_generation_seconds: Float64,
        mean_search_seconds: Float64,
        mean_candidate_hit_count: Float64,
        mean_candidate_recall_at_final_k: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        stage1_vector_count: Int,
        stage1_token_count: Int,
        stage1_byte_size: Int,
        stage1_bytes_per_document: Float64,
        stage1_bytes_per_vector: Float64,
        mean_stage1_graph_visited_vertex_count: Float64,
        mean_stage1_graph_expanded_edge_count: Float64,
        mean_stage1_graph_visited_cluster_count: Float64,
        mean_stage1_graph_entry_point_count: Float64,
        mean_stage1_graph_max_frontier_size: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.candidate_generator = candidate_generator.copy()
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_vector_budget = query_vector_budget
        self.requested_stage1_vector_budget = requested_stage1_vector_budget
        self.posting_cap = posting_cap
        self.query_count = query_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.vector_dim = vector_dim
        self.mean_candidate_generation_seconds = mean_candidate_generation_seconds
        self.mean_search_seconds = mean_search_seconds
        self.mean_candidate_hit_count = mean_candidate_hit_count
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.stage1_vector_count = stage1_vector_count
        self.stage1_token_count = stage1_token_count
        self.stage1_byte_size = stage1_byte_size
        self.stage1_bytes_per_document = stage1_bytes_per_document
        self.stage1_bytes_per_vector = stage1_bytes_per_vector
        self.mean_stage1_graph_visited_vertex_count = (
            mean_stage1_graph_visited_vertex_count
        )
        self.mean_stage1_graph_expanded_edge_count = (
            mean_stage1_graph_expanded_edge_count
        )
        self.mean_stage1_graph_visited_cluster_count = (
            mean_stage1_graph_visited_cluster_count
        )
        self.mean_stage1_graph_entry_point_count = (
            mean_stage1_graph_entry_point_count
        )
        self.mean_stage1_graph_max_frontier_size = (
            mean_stage1_graph_max_frontier_size
        )


def density_bytes_per_document(byte_size: Int, document_count: Int) -> Float64:
    if document_count == 0:
        return 0.0

    return Float64(byte_size) / Float64(document_count)


def density_bytes_per_vector(byte_size: Int, vector_count: Int) -> Float64:
    if vector_count == 0:
        return 0.0

    return Float64(byte_size) / Float64(vector_count)


def build_faithfulness_frontier_summary_for_plan(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    query_vector_budget: Int,
    requested_stage1_vector_budget: Int,
    posting_cap: Int,
) raises -> FaithfulnessFrontierSummary:
    var task = stored_task.task.copy()
    if len(task.queries) == 0:
        raise Error("faithfulness frontier benchmark requires at least one judged query")

    var candidate_hit_total = 0.0
    var candidate_recall_total = 0.0
    var ndcg_total = 0.0
    var reciprocal_rank_total = 0.0
    var recall_total = 0.0
    var success_total = 0.0
    var stage1_vector_count = 0
    var stage1_token_count = 0
    var stage1_byte_size = 0
    var stage1_graph_visited_vertex_total = 0.0
    var stage1_graph_expanded_edge_total = 0.0
    var stage1_graph_visited_cluster_total = 0.0
    var stage1_graph_entry_point_total = 0.0
    var stage1_graph_max_frontier_total = 0.0
    var query_index = 0

    def candidate_once() capturing raises:
        bench_compiler.keep(
            candidate_generation_for_plan(
                backend,
                truncate_query_to_vector_budget(
                    task.queries[query_index].query,
                    query_vector_budget,
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
    query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(
            search_collection_for_plan(
                backend,
                truncate_query_to_vector_budget(
                    task.queries[query_index].query,
                    query_vector_budget,
                ),
                snapshot,
                plan,
                query_text=judged_query_text_for_plan(
                    plan,
                    task.queries[query_index].description,
                ),
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var search_report = run[search_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )

    for judged_query in task.queries:
        var budgeted_query = truncate_query_to_vector_budget(
            judged_query.query,
            query_vector_budget,
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

        candidate_hit_total += Float64(len(explain.candidate_set.hits))
        candidate_recall_total += Float64(explain.candidate_recall_at_final_k)
        ndcg_total += Float64(query_evaluation.ndcg_at_k)
        reciprocal_rank_total += Float64(
            query_evaluation.reciprocal_rank_at_k
        )
        recall_total += Float64(query_evaluation.recall_at_k)
        success_total += Float64(query_evaluation.success_at_k)
        stage1_graph_visited_vertex_total += Float64(
            explain.candidate_stage.graph_search_counters.visited_vertex_count
        )
        stage1_graph_expanded_edge_total += Float64(
            explain.candidate_stage.graph_search_counters.expanded_edge_count
        )
        stage1_graph_visited_cluster_total += Float64(
            explain.candidate_stage.graph_search_counters.visited_cluster_count
        )
        stage1_graph_entry_point_total += Float64(
            explain.candidate_stage.graph_search_counters.entry_point_count
        )
        stage1_graph_max_frontier_total += Float64(
            explain.candidate_stage.graph_search_counters.max_frontier_size
        )

        if stage1_vector_count == 0 and stage1_token_count == 0 and stage1_byte_size == 0:
            stage1_vector_count = explain.candidate_stage.vector_count
            stage1_token_count = explain.candidate_stage.token_count
            stage1_byte_size = explain.candidate_stage.byte_size

    var query_count = len(task.queries)
    return FaithfulnessFrontierSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        plan.candidate_generator,
        task.k,
        plan.candidate_budget.candidate_k,
        query_vector_budget,
        requested_stage1_vector_budget,
        posting_cap,
        query_count,
        snapshot.snapshot.stats.document_count,
        snapshot.snapshot.stats.token_count,
        snapshot.snapshot.stats.total_vector_count,
        snapshot.collection.vector_dim,
        Float64(candidate_report.mean()),
        Float64(search_report.mean()),
        candidate_hit_total / Float64(query_count),
        candidate_recall_total / Float64(query_count),
        ndcg_total / Float64(query_count),
        reciprocal_rank_total / Float64(query_count),
        recall_total / Float64(query_count),
        success_total / Float64(query_count),
        stage1_vector_count,
        stage1_token_count,
        stage1_byte_size,
        density_bytes_per_document(stage1_byte_size, snapshot.snapshot.stats.document_count),
        density_bytes_per_vector(stage1_byte_size, stage1_vector_count),
        stage1_graph_visited_vertex_total / Float64(query_count),
        stage1_graph_expanded_edge_total / Float64(query_count),
        stage1_graph_visited_cluster_total / Float64(query_count),
        stage1_graph_entry_point_total / Float64(query_count),
        stage1_graph_max_frontier_total / Float64(query_count),
    )


def append_faithfulness_frontier_summary_json(
    mut buffer: String,
    read summary: FaithfulnessFrontierSummary,
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    append_candidate_generator_semantics_json_fields(
        buffer,
        summary.candidate_generator,
    )
    buffer += ","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_vector_budget\":" + String(summary.query_vector_budget) + ","
    buffer += "\"requested_stage1_vector_budget\":"
    buffer += String(summary.requested_stage1_vector_budget) + ","
    buffer += "\"posting_cap\":" + String(summary.posting_cap) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"token_count\":" + String(summary.token_count) + ","
    buffer += "\"vector_count\":" + String(summary.vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"mean_candidate_generation_seconds\":"
    buffer += String(summary.mean_candidate_generation_seconds) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds) + ","
    buffer += "\"mean_candidate_hit_count\":"
    buffer += String(summary.mean_candidate_hit_count) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"stage1_vector_count\":" + String(summary.stage1_vector_count) + ","
    buffer += "\"stage1_token_count\":" + String(summary.stage1_token_count) + ","
    buffer += "\"stage1_byte_size\":" + String(summary.stage1_byte_size) + ","
    buffer += "\"stage1_bytes_per_document\":"
    buffer += String(summary.stage1_bytes_per_document) + ","
    buffer += "\"stage1_bytes_per_vector\":"
    buffer += String(summary.stage1_bytes_per_vector) + ","
    buffer += "\"mean_stage1_graph_visited_vertex_count\":"
    buffer += String(summary.mean_stage1_graph_visited_vertex_count) + ","
    buffer += "\"mean_stage1_graph_expanded_edge_count\":"
    buffer += String(summary.mean_stage1_graph_expanded_edge_count) + ","
    buffer += "\"mean_stage1_graph_visited_cluster_count\":"
    buffer += String(summary.mean_stage1_graph_visited_cluster_count) + ","
    buffer += "\"mean_stage1_graph_entry_point_count\":"
    buffer += String(summary.mean_stage1_graph_entry_point_count) + ","
    buffer += "\"mean_stage1_graph_max_frontier_size\":"
    buffer += String(summary.mean_stage1_graph_max_frontier_size)
    buffer += "}"


def faithfulness_frontier_summary_json(
    read summary: FaithfulnessFrontierSummary
) -> String:
    var buffer = String()
    append_faithfulness_frontier_summary_json(buffer, summary)
    return buffer^


def faithfulness_frontier_summaries_json(
    read summaries: List[FaithfulnessFrontierSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_faithfulness_frontier_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
