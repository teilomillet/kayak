from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.eval import JudgedTask, TaskEvaluation, evaluate_query_hits, choose_primary_value
from kayak.numeric import MetricScalar, zero_metric_scalar
from kayak.planning import (
    SearchPlan,
    explain_collection_search,
    exact_full_scan_clause_text_search_plan,
    exact_full_scan_search_plan,
    final_hits_to_search_hits,
    search_collection_for_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.search import SearchHit, search_exact
from kayak.storage import StoredJudgedTask, StoredPackedIndex
from kayak.text import DocumentTextCorpus
from kayak.verifier import default_clause_text_rerank_config, rerank_hits_clause_text

from .json_common import append_json_string_list, json_escape
from .materialized_artifact_families import materialized_artifact_families
from .query_text_support import judged_query_text_for_plan
from .search_plan_semantics_json import append_search_plan_semantics_json_fields


comptime CEILING_BENCH_MIN_SECONDS = 0.05
comptime CEILING_BENCH_MAX_SECONDS = 0.25
comptime CEILING_BENCH_MAX_ITERS = 200

comptime CEILING_COMPARISON_ROLE_REFERENCE_ANCHOR = "reference_anchor"
comptime CEILING_COMPARISON_ROLE_CANDIDATE_PLAN = "candidate_plan"
comptime CEILING_COMPARISON_ROLE_LOCAL_STRONGER_CEILING = "local_stronger_ceiling"

comptime CEILING_EXECUTION_PATH_EXACT_LATE_INTERACTION = "exact_late_interaction"
comptime CEILING_EXECUTION_PATH_PLANNED_STAGE_AWARE = "planned_stage_aware"
comptime CEILING_EXECUTION_PATH_EXACT_THEN_CLAUSE_TEXT_RERANK = (
    "exact_then_clause_text_rerank"
)


struct CeilingComparisonSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var method_kind: String
    var comparison_role: String
    var execution_path_kind: String
    var plan: SearchPlan
    var stage2_reference_materialized_artifact_families: List[String]
    var stage3_verifier_materialized_artifact_families: List[String]
    var primary_metric: String
    var primary_value: Float64
    var mean_candidate_recall_at_final_k: Float64
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
        var method_kind: String,
        var comparison_role: String,
        var execution_path_kind: String,
        plan: SearchPlan,
        read stage2_reference_materialized_artifact_families: List[String],
        read stage3_verifier_materialized_artifact_families: List[String],
        var primary_metric: String,
        primary_value: Float64,
        mean_candidate_recall_at_final_k: Float64,
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
        self.method_kind = method_kind^
        self.comparison_role = comparison_role^
        self.execution_path_kind = execution_path_kind^
        self.plan = plan.copy()
        self.stage2_reference_materialized_artifact_families = (
            stage2_reference_materialized_artifact_families.copy()
        )
        self.stage3_verifier_materialized_artifact_families = (
            stage3_verifier_materialized_artifact_families.copy()
        )
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.mean_search_seconds = mean_search_seconds


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


def build_task_evaluation(
    read task: JudgedTask,
    primary_metric: String,
    ndcg_total: MetricScalar,
    reciprocal_rank_total: MetricScalar,
    recall_total: MetricScalar,
    success_total: MetricScalar,
) raises -> TaskEvaluation:
    var query_count = len(task.queries)
    var mean_ndcg_at_k = ndcg_total / MetricScalar(query_count)
    var mean_reciprocal_rank = reciprocal_rank_total / MetricScalar(query_count)
    var mean_recall_at_k = recall_total / MetricScalar(query_count)
    var success_rate_at_k = success_total / MetricScalar(query_count)

    return TaskEvaluation(
        task.family.copy(),
        task.slice_name.copy(),
        primary_metric.copy(),
        choose_primary_value(
            primary_metric,
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
def build_exact_full_scan_ceiling_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
) raises -> CeilingComparisonSummary:
    var task = stored_task.task.copy()
    var query_index = 0
    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()

    for judged_query in task.queries:
        var hits = search_exact(backend, judged_query.query, stored_index.index, task.k)
        var query_evaluation = evaluate_query_hits(
            judged_query, hits, task.k, task.primary_metric
        )
        ndcg_total += query_evaluation.ndcg_at_k
        reciprocal_rank_total += query_evaluation.reciprocal_rank_at_k
        recall_total += query_evaluation.recall_at_k
        success_total += query_evaluation.success_at_k

    def search_once() capturing raises:
        bench_compiler.keep(
            search_exact(
                backend,
                task.queries[query_index].query,
                stored_index.index,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = run[search_once](
        num_warmup_iters=1,
        max_iters=CEILING_BENCH_MAX_ITERS,
        min_runtime_secs=CEILING_BENCH_MIN_SECONDS,
        max_runtime_secs=CEILING_BENCH_MAX_SECONDS,
    )
    var evaluation = build_task_evaluation(
        task,
        task.primary_metric,
        ndcg_total,
        reciprocal_rank_total,
        recall_total,
        success_total,
    )

    return CeilingComparisonSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        "exact_full_scan",
        CEILING_COMPARISON_ROLE_REFERENCE_ANCHOR,
        CEILING_EXECUTION_PATH_EXACT_LATE_INTERACTION,
        exact_full_scan_search_plan(task.k, task.k),
        [],
        [],
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        1.0,
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        Float64(report.mean()),
    )


def build_exact_clause_text_ceiling_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
    read document_texts: DocumentTextCorpus,
    candidate_k: Int,
) raises -> CeilingComparisonSummary:
    var task = stored_task.task.copy()
    var config = default_clause_text_rerank_config()
    var query_index = 0
    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()

    for judged_query in task.queries:
        var reference_hits = search_exact(
            backend, judged_query.query, stored_index.index, task.k
        )
        var candidate_hits = search_exact(
            backend, judged_query.query, stored_index.index, candidate_k
        )
        var reranked_hits = rerank_hits_clause_text(
            judged_query.description,
            candidate_hits,
            document_texts,
            task.k,
            config,
        )
        var query_evaluation = evaluate_query_hits(
            judged_query, reranked_hits, task.k, task.primary_metric
        )
        ndcg_total += query_evaluation.ndcg_at_k
        reciprocal_rank_total += query_evaluation.reciprocal_rank_at_k
        recall_total += query_evaluation.recall_at_k
        success_total += query_evaluation.success_at_k
        _ = reference_recall_at_k(reference_hits, candidate_hits, task.k)
    def search_once() capturing raises:
        var candidate_hits = search_exact(
            backend,
            task.queries[query_index].query,
            stored_index.index,
            candidate_k,
        )
        bench_compiler.keep(
            rerank_hits_clause_text(
                task.queries[query_index].description,
                candidate_hits,
                document_texts,
                task.k,
                config,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = run[search_once](
        num_warmup_iters=1,
        max_iters=CEILING_BENCH_MAX_ITERS,
        min_runtime_secs=CEILING_BENCH_MIN_SECONDS,
        max_runtime_secs=CEILING_BENCH_MAX_SECONDS,
    )
    var evaluation = build_task_evaluation(
        task,
        task.primary_metric,
        ndcg_total,
        reciprocal_rank_total,
        recall_total,
        success_total,
    )

    return CeilingComparisonSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        "exact_clause_text_ceiling",
        CEILING_COMPARISON_ROLE_LOCAL_STRONGER_CEILING,
        CEILING_EXECUTION_PATH_EXACT_THEN_CLAUSE_TEXT_RERANK,
        exact_full_scan_clause_text_search_plan(task.k, candidate_k),
        [],
        ["document_text"],
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        1.0,
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        Float64(report.mean()),
    )


def build_stage_aware_ceiling_summary_for_plan(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> CeilingComparisonSummary:
    var task = stored_task.task.copy()
    var query_index = 0
    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()
    var candidate_recall_total = 0.0
    var representative_stage2_reference_materialized_artifact_families = List[String]()
    var representative_stage3_verifier_materialized_artifact_families = List[String]()

    for judged_query in task.queries:
        var query_text = judged_query_text_for_plan(
            plan,
            judged_query.description,
        )
        var explain = explain_collection_search(
            backend,
            judged_query.query,
            snapshot,
            plan,
            query_text=query_text,
        )
        if len(representative_stage2_reference_materialized_artifact_families) == 0:
            representative_stage2_reference_materialized_artifact_families = (
                materialized_artifact_families(
                    explain.stage2.materialized_artifacts
                )
            )
        if len(representative_stage3_verifier_materialized_artifact_families) == 0:
            representative_stage3_verifier_materialized_artifact_families = (
                materialized_artifact_families(
                    explain.stage3_verifier.materialized_artifacts
                )
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
        num_warmup_iters=1,
        max_iters=CEILING_BENCH_MAX_ITERS,
        min_runtime_secs=CEILING_BENCH_MIN_SECONDS,
        max_runtime_secs=CEILING_BENCH_MAX_SECONDS,
    )
    var evaluation = build_task_evaluation(
        task,
        task.primary_metric,
        ndcg_total,
        reciprocal_rank_total,
        recall_total,
        success_total,
    )

    return CeilingComparisonSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        "stage_aware",
        CEILING_COMPARISON_ROLE_CANDIDATE_PLAN,
        CEILING_EXECUTION_PATH_PLANNED_STAGE_AWARE,
        plan,
        representative_stage2_reference_materialized_artifact_families^,
        representative_stage3_verifier_materialized_artifact_families^,
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        candidate_recall_total / Float64(len(task.queries)),
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        Float64(report.mean()),
    )


def append_ceiling_comparison_summary_json(
    mut buffer: String, read summary: CeilingComparisonSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"method_kind\":\"" + json_escape(summary.method_kind) + "\","
    buffer += "\"comparison_role\":\""
    buffer += json_escape(summary.comparison_role) + "\","
    buffer += "\"execution_path_kind\":\""
    buffer += json_escape(summary.execution_path_kind) + "\","
    append_search_plan_semantics_json_fields(buffer, summary.plan)
    buffer += ","
    buffer += "\"stage2_reference_materialized_artifact_families\":"
    append_json_string_list(
        buffer,
        summary.stage2_reference_materialized_artifact_families,
    )
    buffer += ","
    buffer += "\"stage3_verifier_materialized_artifact_families\":"
    append_json_string_list(
        buffer,
        summary.stage3_verifier_materialized_artifact_families,
    )
    buffer += ","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds)
    buffer += "}"


def ceiling_comparison_summary_json(
    read summary: CeilingComparisonSummary
) -> String:
    var buffer = String()
    append_ceiling_comparison_summary_json(buffer, summary)
    return buffer^


def ceiling_comparison_summaries_json(
    read summaries: List[CeilingComparisonSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_ceiling_comparison_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
