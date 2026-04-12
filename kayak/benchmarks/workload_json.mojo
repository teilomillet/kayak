from std.collections import List

from .json_common import json_escape
from .workload_profile import WorkloadProfile


struct WorkloadBenchmarkSummary(Copyable):
    var family: String
    var slice_name: String
    var why: String
    var document_count: Int
    var document_vector_count: Int
    var query_vector_count: Int
    var vector_dim: Int
    var top_k: Int
    var mean_search_seconds: Float64

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var why: String,
        document_count: Int,
        document_vector_count: Int,
        query_vector_count: Int,
        vector_dim: Int,
        top_k: Int,
        mean_search_seconds: Float64,
    ):
        self.family = family^
        self.slice_name = slice_name^
        self.why = why^
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.query_vector_count = query_vector_count
        self.vector_dim = vector_dim
        self.top_k = top_k
        self.mean_search_seconds = mean_search_seconds


def build_workload_benchmark_summary(
    read profile: WorkloadProfile, mean_search_seconds: Float64
) -> WorkloadBenchmarkSummary:
    return WorkloadBenchmarkSummary(
        profile.family.copy(),
        profile.slice_name.copy(),
        profile.why.copy(),
        profile.document_count,
        profile.document_vector_count,
        profile.query_vector_count,
        profile.vector_dim,
        profile.top_k,
        mean_search_seconds,
    )


def append_workload_benchmark_summary_json(
    mut buffer: String, read summary: WorkloadBenchmarkSummary
):
    buffer += "{"
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"why\":\"" + json_escape(summary.why) + "\","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"document_vector_count\":"
    buffer += String(summary.document_vector_count) + ","
    buffer += "\"query_vector_count\":" + String(summary.query_vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"top_k\":" + String(summary.top_k) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds)
    buffer += "}"


def workload_benchmark_summaries_json(
    read summaries: List[WorkloadBenchmarkSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","

        append_workload_benchmark_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
