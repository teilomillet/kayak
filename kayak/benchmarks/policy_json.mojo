from std.collections import List

from kayak.eval import JudgedTask

from .json_common import json_escape


struct BackendPolicyBenchmarkSummary(Copyable):
    var dataset_name: String
    var slice_name: String
    var task_source: String
    var index_source: String
    var backend_name: String
    var query_count: Int
    var document_count: Int
    var nominal_query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var top_k: Int
    var mean_search_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var task_source: String,
        var index_source: String,
        var backend_name: String,
        query_count: Int,
        document_count: Int,
        nominal_query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        top_k: Int,
        mean_search_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.task_source = task_source^
        self.index_source = index_source^
        self.backend_name = backend_name^
        self.query_count = query_count
        self.document_count = document_count
        self.nominal_query_vector_count = nominal_query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.top_k = top_k
        self.mean_search_seconds = mean_search_seconds


def build_backend_policy_benchmark_summary(
    dataset_name: String,
    read task: JudgedTask,
    task_source: String,
    index_source: String,
    backend_name: String,
    mean_search_seconds: Float64,
) -> BackendPolicyBenchmarkSummary:
    return BackendPolicyBenchmarkSummary(
        dataset_name.copy(),
        task.slice_name.copy(),
        task_source.copy(),
        index_source.copy(),
        backend_name.copy(),
        len(task.queries),
        len(task.documents),
        task.nominal_query_vector_count,
        task.nominal_document_vector_count,
        task.vector_dim,
        task.k,
        mean_search_seconds,
    )


def append_backend_policy_benchmark_summary_json(
    mut buffer: String, read summary: BackendPolicyBenchmarkSummary
):
    buffer += "{"
    buffer += "\"dataset_name\":\"" + json_escape(summary.dataset_name) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"task_source\":\"" + json_escape(summary.task_source) + "\","
    buffer += "\"index_source\":\"" + json_escape(summary.index_source) + "\","
    buffer += "\"backend_name\":\"" + json_escape(summary.backend_name) + "\","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"nominal_query_vector_count\":"
    buffer += String(summary.nominal_query_vector_count) + ","
    buffer += "\"nominal_document_vector_count\":"
    buffer += String(summary.nominal_document_vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"top_k\":" + String(summary.top_k) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds)
    buffer += "}"


def backend_policy_benchmark_summaries_json(
    read summaries: List[BackendPolicyBenchmarkSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","

        append_backend_policy_benchmark_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
