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

from .json_common import append_json_string_list, json_escape
from .materialized_artifact_families import materialized_artifact_families
from .query_text_support import judged_query_text_for_plan


struct StageAwareSearchSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var collection_id: String
    var snapshot_id: String
    var candidate_generator_kind: String
    var stage2_kind: String
    var stage2_family: String
    var stage2_requires_query_text: Bool
    var stage2_materialized_artifact_families: List[String]
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
    var nominal_query_vector_count: Int
    var nominal_document_vector_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int
    var bytes_per_document: Float64
    var bytes_per_vector: Float64
    var candidate_stage_document_count: Int
    var candidate_stage_token_count: Int
    var candidate_stage_vector_count: Int
    var candidate_stage_byte_size: Int
    var candidate_stage_bytes_per_document: Float64
    var candidate_stage_bytes_per_vector: Float64
    var candidate_stage_tracks_graph_search: Bool
    var mean_candidate_stage_graph_visited_vertex_count: Float64
    var mean_candidate_stage_graph_expanded_edge_count: Float64
    var mean_candidate_stage_graph_visited_cluster_count: Float64
    var mean_candidate_stage_graph_entry_point_count: Float64
    var mean_candidate_stage_graph_max_frontier_size: Float64
    var stage2_document_count: Int
    var stage2_token_count: Int
    var stage2_vector_count: Int
    var stage2_byte_size: Int
    var stage2_bytes_per_document: Float64
    var stage2_bytes_per_vector: Float64
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
        var stage2_kind: String,
        var stage2_family: String,
        stage2_requires_query_text: Bool,
        var stage2_materialized_artifact_families: List[String],
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
        nominal_query_vector_count: Int,
        nominal_document_vector_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
        bytes_per_document: Float64,
        bytes_per_vector: Float64,
        candidate_stage_document_count: Int,
        candidate_stage_token_count: Int,
        candidate_stage_vector_count: Int,
        candidate_stage_byte_size: Int,
        candidate_stage_bytes_per_document: Float64,
        candidate_stage_bytes_per_vector: Float64,
        candidate_stage_tracks_graph_search: Bool,
        mean_candidate_stage_graph_visited_vertex_count: Float64,
        mean_candidate_stage_graph_expanded_edge_count: Float64,
        mean_candidate_stage_graph_visited_cluster_count: Float64,
        mean_candidate_stage_graph_entry_point_count: Float64,
        mean_candidate_stage_graph_max_frontier_size: Float64,
        stage2_document_count: Int,
        stage2_token_count: Int,
        stage2_vector_count: Int,
        stage2_byte_size: Int,
        stage2_bytes_per_document: Float64,
        stage2_bytes_per_vector: Float64,
        vector_dim: Int,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.candidate_generator_kind = candidate_generator_kind^
        self.stage2_kind = stage2_kind^
        self.stage2_family = stage2_family^
        self.stage2_requires_query_text = stage2_requires_query_text
        self.stage2_materialized_artifact_families = (
            stage2_materialized_artifact_families^
        )
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
        self.nominal_query_vector_count = nominal_query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
        self.bytes_per_document = bytes_per_document
        self.bytes_per_vector = bytes_per_vector
        self.candidate_stage_document_count = candidate_stage_document_count
        self.candidate_stage_token_count = candidate_stage_token_count
        self.candidate_stage_vector_count = candidate_stage_vector_count
        self.candidate_stage_byte_size = candidate_stage_byte_size
        self.candidate_stage_bytes_per_document = (
            candidate_stage_bytes_per_document
        )
        self.candidate_stage_bytes_per_vector = candidate_stage_bytes_per_vector
        self.candidate_stage_tracks_graph_search = candidate_stage_tracks_graph_search
        self.mean_candidate_stage_graph_visited_vertex_count = (
            mean_candidate_stage_graph_visited_vertex_count
        )
        self.mean_candidate_stage_graph_expanded_edge_count = (
            mean_candidate_stage_graph_expanded_edge_count
        )
        self.mean_candidate_stage_graph_visited_cluster_count = (
            mean_candidate_stage_graph_visited_cluster_count
        )
        self.mean_candidate_stage_graph_entry_point_count = (
            mean_candidate_stage_graph_entry_point_count
        )
        self.mean_candidate_stage_graph_max_frontier_size = (
            mean_candidate_stage_graph_max_frontier_size
        )
        self.stage2_document_count = stage2_document_count
        self.stage2_token_count = stage2_token_count
        self.stage2_vector_count = stage2_vector_count
        self.stage2_byte_size = stage2_byte_size
        self.stage2_bytes_per_document = stage2_bytes_per_document
        self.stage2_bytes_per_vector = stage2_bytes_per_vector
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
    var stage2_materialized_artifact_families: List[String],
    candidate_stage_document_count: Int,
    candidate_stage_token_count: Int,
    candidate_stage_vector_count: Int,
    candidate_stage_byte_size: Int,
    candidate_stage_tracks_graph_search: Bool,
    mean_candidate_stage_graph_visited_vertex_count: Float64,
    mean_candidate_stage_graph_expanded_edge_count: Float64,
    mean_candidate_stage_graph_visited_cluster_count: Float64,
    mean_candidate_stage_graph_entry_point_count: Float64,
    mean_candidate_stage_graph_max_frontier_size: Float64,
    stage2_document_count: Int,
    stage2_token_count: Int,
    stage2_vector_count: Int,
    stage2_byte_size: Int,
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
        plan.stage2_operator.kind.copy(),
        plan.stage2_operator.family.copy(),
        plan.stage2_operator.requires_query_text,
        stage2_materialized_artifact_families^,
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
        task.nominal_query_vector_count,
        task.nominal_document_vector_count,
        stats.document_count,
        stats.token_count,
        stats.total_vector_count,
        stats.byte_size,
        density_bytes_per_document(stats.byte_size, stats.document_count),
        density_bytes_per_vector(stats.byte_size, stats.total_vector_count),
        candidate_stage_document_count,
        candidate_stage_token_count,
        candidate_stage_vector_count,
        candidate_stage_byte_size,
        density_bytes_per_document(
            candidate_stage_byte_size, candidate_stage_document_count
        ),
        density_bytes_per_vector(
            candidate_stage_byte_size, candidate_stage_vector_count
        ),
        candidate_stage_tracks_graph_search,
        mean_candidate_stage_graph_visited_vertex_count,
        mean_candidate_stage_graph_expanded_edge_count,
        mean_candidate_stage_graph_visited_cluster_count,
        mean_candidate_stage_graph_entry_point_count,
        mean_candidate_stage_graph_max_frontier_size,
        stage2_document_count,
        stage2_token_count,
        stage2_vector_count,
        stage2_byte_size,
        density_bytes_per_document(
            stage2_byte_size, stage2_document_count
        ),
        density_bytes_per_vector(stage2_byte_size, stage2_vector_count),
        snapshot.collection.vector_dim,
    )
def build_stage_aware_search_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> StageAwareSearchSummary:
    var task = stored_task.task.copy()
    if len(task.queries) == 0:
        raise Error("stage-aware benchmark requires at least one judged query")

    var first_query = task.queries[0].copy()
    var first_query_text = judged_query_text_for_plan(
        plan,
        first_query.description,
    )
    var first_final_hits = search_collection_for_plan(
        backend,
        first_query.query,
        snapshot,
        plan,
        query_text=first_query_text,
    )
    var first_query_evaluation = evaluate_query_hits(
        first_query,
        final_hits_to_search_hits(first_final_hits),
        task.k,
        task.primary_metric,
    )
    var representative_explain = explain_collection_search(
        backend,
        first_query.query,
        snapshot,
        plan,
        query_text=first_query_text,
    )

    var primary_total = Float64(first_query_evaluation.primary_value)
    var ndcg_total = Float64(first_query_evaluation.ndcg_at_k)
    var reciprocal_rank_total = Float64(
        first_query_evaluation.reciprocal_rank_at_k
    )
    var recall_total = Float64(first_query_evaluation.recall_at_k)
    var success_total = Float64(first_query_evaluation.success_at_k)
    var candidate_recall_total = Float64(
        representative_explain.candidate_recall_at_final_k
    )
    var candidate_stage_graph_visited_vertex_total = Float64(
        representative_explain.candidate_stage.graph_search_counters.visited_vertex_count
    )
    var candidate_stage_graph_expanded_edge_total = Float64(
        representative_explain.candidate_stage.graph_search_counters.expanded_edge_count
    )
    var candidate_stage_graph_visited_cluster_total = Float64(
        representative_explain.candidate_stage.graph_search_counters.visited_cluster_count
    )
    var candidate_stage_graph_entry_point_total = Float64(
        representative_explain.candidate_stage.graph_search_counters.entry_point_count
    )
    var candidate_stage_graph_max_frontier_total = Float64(
        representative_explain.candidate_stage.graph_search_counters.max_frontier_size
    )

    for query_index in range(1, len(task.queries)):
        var judged_query = task.queries[query_index].copy()
        var query_text = judged_query_text_for_plan(
            plan,
            judged_query.description,
        )
        var final_hits = search_collection_for_plan(
            backend,
            judged_query.query,
            snapshot,
            plan,
            query_text=query_text,
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
            query_text=query_text,
        )
        candidate_recall_total += Float64(explain.candidate_recall_at_final_k)
        candidate_stage_graph_visited_vertex_total += Float64(
            explain.candidate_stage.graph_search_counters.visited_vertex_count
        )
        candidate_stage_graph_expanded_edge_total += Float64(
            explain.candidate_stage.graph_search_counters.expanded_edge_count
        )
        candidate_stage_graph_visited_cluster_total += Float64(
            explain.candidate_stage.graph_search_counters.visited_cluster_count
        )
        candidate_stage_graph_entry_point_total += Float64(
            explain.candidate_stage.graph_search_counters.entry_point_count
        )
        candidate_stage_graph_max_frontier_total += Float64(
            explain.candidate_stage.graph_search_counters.max_frontier_size
        )

    var query_count = len(task.queries)
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(
            search_collection_for_plan(
                backend,
                task.queries[query_index].query,
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
        materialized_artifact_families(
            representative_explain.stage2.materialized_artifacts
        ),
        representative_explain.candidate_stage.document_count,
        representative_explain.candidate_stage.token_count,
        representative_explain.candidate_stage.vector_count,
        representative_explain.candidate_stage.byte_size,
        representative_explain.candidate_stage.tracks_graph_search,
        candidate_stage_graph_visited_vertex_total / Float64(query_count),
        candidate_stage_graph_expanded_edge_total / Float64(query_count),
        candidate_stage_graph_visited_cluster_total / Float64(query_count),
        candidate_stage_graph_entry_point_total / Float64(query_count),
        candidate_stage_graph_max_frontier_total / Float64(query_count),
        representative_explain.stage2.document_count,
        representative_explain.stage2.token_count,
        representative_explain.stage2.vector_count,
        representative_explain.stage2.byte_size,
        primary_total / Float64(query_count),
        ndcg_total / Float64(query_count),
        reciprocal_rank_total / Float64(query_count),
        recall_total / Float64(query_count),
        success_total / Float64(query_count),
        candidate_recall_total / Float64(query_count),
        Float64(report.mean()),
    )


def build_stage_aware_search_summaries_for_plans(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plans: List[SearchPlan],
) raises -> List[StageAwareSearchSummary]:
    var summaries = List[StageAwareSearchSummary]()
    for plan in plans:
        summaries.append(
            build_stage_aware_search_summary(
                backend,
                stored_task,
                snapshot,
                plan,
            )
        )
    return summaries^


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
    buffer += "\"stage2_kind\":\""
    buffer += json_escape(summary.stage2_kind) + "\","
    buffer += "\"stage2_family\":\""
    buffer += json_escape(summary.stage2_family) + "\","
    buffer += "\"stage2_requires_query_text\":"
    if summary.stage2_requires_query_text:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"stage2_materialized_artifact_families\":"
    append_json_string_list(
        buffer,
        summary.stage2_materialized_artifact_families,
    )
    buffer += ","
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
    buffer += "\"nominal_query_vector_count\":"
    buffer += String(summary.nominal_query_vector_count) + ","
    buffer += "\"nominal_document_vector_count\":"
    buffer += String(summary.nominal_document_vector_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"token_count\":" + String(summary.token_count) + ","
    buffer += "\"vector_count\":" + String(summary.vector_count) + ","
    buffer += "\"byte_size\":" + String(summary.byte_size) + ","
    buffer += "\"bytes_per_document\":"
    buffer += String(summary.bytes_per_document) + ","
    buffer += "\"bytes_per_vector\":" + String(summary.bytes_per_vector) + ","
    buffer += "\"candidate_stage_document_count\":"
    buffer += String(summary.candidate_stage_document_count) + ","
    buffer += "\"candidate_stage_token_count\":"
    buffer += String(summary.candidate_stage_token_count) + ","
    buffer += "\"candidate_stage_vector_count\":"
    buffer += String(summary.candidate_stage_vector_count) + ","
    buffer += "\"candidate_stage_byte_size\":"
    buffer += String(summary.candidate_stage_byte_size) + ","
    buffer += "\"candidate_stage_bytes_per_document\":"
    buffer += String(summary.candidate_stage_bytes_per_document) + ","
    buffer += "\"candidate_stage_bytes_per_vector\":"
    buffer += String(summary.candidate_stage_bytes_per_vector) + ","
    buffer += "\"candidate_stage_tracks_graph_search\":"
    if summary.candidate_stage_tracks_graph_search:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"mean_candidate_stage_graph_visited_vertex_count\":"
    buffer += String(summary.mean_candidate_stage_graph_visited_vertex_count) + ","
    buffer += "\"mean_candidate_stage_graph_expanded_edge_count\":"
    buffer += String(summary.mean_candidate_stage_graph_expanded_edge_count) + ","
    buffer += "\"mean_candidate_stage_graph_visited_cluster_count\":"
    buffer += String(summary.mean_candidate_stage_graph_visited_cluster_count) + ","
    buffer += "\"mean_candidate_stage_graph_entry_point_count\":"
    buffer += String(summary.mean_candidate_stage_graph_entry_point_count) + ","
    buffer += "\"mean_candidate_stage_graph_max_frontier_size\":"
    buffer += String(summary.mean_candidate_stage_graph_max_frontier_size) + ","
    buffer += "\"stage2_document_count\":"
    buffer += String(summary.stage2_document_count) + ","
    buffer += "\"stage2_token_count\":"
    buffer += String(summary.stage2_token_count) + ","
    buffer += "\"stage2_vector_count\":"
    buffer += String(summary.stage2_vector_count) + ","
    buffer += "\"stage2_byte_size\":"
    buffer += String(summary.stage2_byte_size) + ","
    buffer += "\"stage2_bytes_per_document\":"
    buffer += String(summary.stage2_bytes_per_document) + ","
    buffer += "\"stage2_bytes_per_vector\":"
    buffer += String(summary.stage2_bytes_per_vector) + ","
    buffer += "\"exact_stage_document_count\":"
    buffer += String(summary.stage2_document_count) + ","
    buffer += "\"exact_stage_token_count\":"
    buffer += String(summary.stage2_token_count) + ","
    buffer += "\"exact_stage_vector_count\":"
    buffer += String(summary.stage2_vector_count) + ","
    buffer += "\"exact_stage_byte_size\":"
    buffer += String(summary.stage2_byte_size) + ","
    buffer += "\"exact_stage_bytes_per_document\":"
    buffer += String(summary.stage2_bytes_per_document) + ","
    buffer += "\"exact_stage_bytes_per_vector\":"
    buffer += String(summary.stage2_bytes_per_vector) + ","
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
