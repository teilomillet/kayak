from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.eval import JudgedTask, TaskEvaluation, evaluate_task
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import StoredJudgedTask, StoredPackedIndex

from .json_common import json_escape


struct RealSliceBenchmarkSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var task_source: String
    var index_source: String
    var primary_metric: String
    var primary_value: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var mean_search_seconds: Float64
    var k: Int
    var query_count: Int
    var document_count: Int
    var nominal_query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var task_source: String,
        var index_source: String,
        var primary_metric: String,
        primary_value: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        mean_search_seconds: Float64,
        k: Int,
        query_count: Int,
        document_count: Int,
        nominal_query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.task_source = task_source^
        self.index_source = index_source^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.mean_search_seconds = mean_search_seconds
        self.k = k
        self.query_count = query_count
        self.document_count = document_count
        self.nominal_query_vector_count = nominal_query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def build_real_slice_benchmark_summary_from_measurement(
    dataset_id: String,
    model_name: String,
    read task: JudgedTask,
    task_source: String,
    index_source: String,
    read evaluation: TaskEvaluation,
    mean_search_seconds: Float64,
) -> RealSliceBenchmarkSummary:
    return RealSliceBenchmarkSummary(
        dataset_id.copy(),
        model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        task_source.copy(),
        index_source.copy(),
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        mean_search_seconds,
        task.k,
        len(task.queries),
        len(task.documents),
        task.nominal_query_vector_count,
        task.nominal_document_vector_count,
        task.vector_dim,
    )


def build_real_slice_benchmark_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
    loaded_task_from_storage: Bool,
    loaded_index_from_storage: Bool,
) raises -> RealSliceBenchmarkSummary:
    var task = stored_task.task.copy()
    var index = stored_index.index.copy()
    var evaluation = evaluate_task(backend, task)
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            search_exact(
                backend,
                task.queries[query_index].query,
                index,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = run[score_once]()

    return build_real_slice_benchmark_summary_from_measurement(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task,
        source_label(loaded_task_from_storage, "colbert_cpu_encode"),
        source_label(loaded_index_from_storage, "pack_documents"),
        evaluation,
        Float64(report.mean()),
    )


def append_real_slice_benchmark_summary_json(
    mut buffer: String, read summary: RealSliceBenchmarkSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"task_source\":\"" + json_escape(summary.task_source) + "\","
    buffer += "\"index_source\":\"" + json_escape(summary.index_source) + "\","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":" + String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds) + ","
    buffer += "\"k\":" + String(summary.k) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"nominal_query_vector_count\":"
    buffer += String(summary.nominal_query_vector_count) + ","
    buffer += "\"nominal_document_vector_count\":"
    buffer += String(summary.nominal_document_vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim)
    buffer += "}"


def real_slice_benchmark_summary_json(read summary: RealSliceBenchmarkSummary) -> String:
    var buffer = String()
    append_real_slice_benchmark_summary_json(buffer, summary)
    return buffer^


def real_slice_benchmark_summaries_json(
    read summaries: List[RealSliceBenchmarkSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","

        append_real_slice_benchmark_summary_json(buffer, summaries[index])
    buffer += "]"
    return buffer^
