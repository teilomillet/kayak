from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.index import (
    LatentProxyIndex,
    build_query_latent_proxy_vector,
)
from kayak.index.latent_proxy import (
    build_query_latent_proxy_vector_multi_block,
    build_query_latent_proxy_vector_single_block,
)
from kayak.interop import load_task_json
from kayak.numeric import VectorScalar
from kayak.storage import load_stored_latent_proxy_index

from .json_common import json_escape


# Owns focused latent-proxy microbenchmarks. It isolates projection cost from
# the document scan so native runtime regressions can be explained with direct
# measurements instead of inference.


comptime LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS = 0.05
comptime LATENT_PROXY_PROFILE_WARMUP_ITERS = 4
comptime LATENT_PROXY_PROFILE_QUERY_MULTIPLIER = 64


struct LatentProxyProjectionProfileSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var query_count: Int
    var document_count: Int
    var projection_block_count: Int
    var single_block_fast_path_available: Bool
    var query_vector_budget: Int
    var vector_dim: Int
    var latent_dim: Int
    var mean_projection_generic_seconds: Float64
    var mean_projection_single_block_seconds: Float64
    var mean_projection_seconds: Float64
    var mean_scan_seconds: Float64
    var mean_projection_plus_scan_seconds: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        query_count: Int,
        document_count: Int,
        projection_block_count: Int,
        single_block_fast_path_available: Bool,
        query_vector_budget: Int,
        vector_dim: Int,
        latent_dim: Int,
        mean_projection_generic_seconds: Float64,
        mean_projection_single_block_seconds: Float64,
        mean_projection_seconds: Float64,
        mean_scan_seconds: Float64,
        mean_projection_plus_scan_seconds: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.query_count = query_count
        self.document_count = document_count
        self.projection_block_count = projection_block_count
        self.single_block_fast_path_available = single_block_fast_path_available
        self.query_vector_budget = query_vector_budget
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.mean_projection_generic_seconds = mean_projection_generic_seconds
        self.mean_projection_single_block_seconds = (
            mean_projection_single_block_seconds
        )
        self.mean_projection_seconds = mean_projection_seconds
        self.mean_scan_seconds = mean_scan_seconds
        self.mean_projection_plus_scan_seconds = mean_projection_plus_scan_seconds


def score_query_proxy_against_index(
    read query_proxy: List[VectorScalar],
    read proxy_index: LatentProxyIndex,
) raises -> Float64:
    var total = Float64(0.0)
    for document_index in range(proxy_index.document_count):
        for dim_index in range(proxy_index.vector_dim):
            total += (
                Float64(query_proxy[dim_index])
                * Float64(proxy_index.proxy_vectors[document_index][dim_index])
            )
    return total


def max_query_vector_budget_for_task(path: String) raises -> Int:
    var task = load_task_json(path)
    var max_budget = 0
    for judged_query in task.queries:
        if judged_query.query.vector_count > max_budget:
            max_budget = judged_query.query.vector_count

    if max_budget <= 0:
        return 1
    return max_budget


def build_latent_proxy_projection_profile_summary(
    dataset_id: String,
    model_name: String,
    task_path: String,
    artifact_root: String,
) raises -> LatentProxyProjectionProfileSummary:
    var task = load_task_json(task_path)
    var stored = load_stored_latent_proxy_index(artifact_root)
    var projection = stored.query_projection.copy()
    var index = stored.index.copy()
    if len(task.queries) == 0:
        raise Error("latent proxy projection profile requires at least one query")

    var query_index = 0
    var benchmark_iters = len(task.queries) * LATENT_PROXY_PROFILE_QUERY_MULTIPLIER

    def projection_once() capturing raises:
        bench_compiler.keep(
            build_query_latent_proxy_vector(
                task.queries[query_index].query,
                projection,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var projection_report = run[projection_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=benchmark_iters,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    query_index = 0

    def projection_generic_once() capturing raises:
        bench_compiler.keep(
            build_query_latent_proxy_vector_multi_block(
                task.queries[query_index].query,
                projection,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var projection_generic_report = run[projection_generic_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=benchmark_iters,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    var projection_single_block_mean = Float64(-1.0)
    if len(projection.blocks) == 1:
        query_index = 0

        def projection_single_block_once() capturing raises:
            bench_compiler.keep(
                build_query_latent_proxy_vector_single_block(
                    task.queries[query_index].query,
                    projection,
                )
            )
            query_index += 1
            if query_index == len(task.queries):
                query_index = 0

        var projection_single_block_report = run[projection_single_block_once](
            num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
            max_iters=benchmark_iters,
            min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
            max_batch_size=1,
        )
        projection_single_block_mean = Float64(projection_single_block_report.mean())

    var projected_queries = List[List[VectorScalar]]()
    for judged_query in task.queries:
        projected_queries.append(
            build_query_latent_proxy_vector(judged_query.query, projection)
        )

    query_index = 0

    def scan_once() capturing raises:
        bench_compiler.keep(
            score_query_proxy_against_index(projected_queries[query_index], index)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var scan_report = run[scan_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=benchmark_iters,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    query_index = 0

    def projection_plus_scan_once() capturing raises:
        var projected = build_query_latent_proxy_vector(
            task.queries[query_index].query,
            projection,
        )
        bench_compiler.keep(score_query_proxy_against_index(projected, index))
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var projection_plus_scan_report = run[projection_plus_scan_once](
        num_warmup_iters=LATENT_PROXY_PROFILE_WARMUP_ITERS,
        max_iters=benchmark_iters,
        min_runtime_secs=LATENT_PROXY_PROFILE_MIN_RUNTIME_SECS,
        max_batch_size=1,
    )

    return LatentProxyProjectionProfileSummary(
        dataset_id.copy(),
        model_name.copy(),
        len(task.queries),
        len(task.documents),
        len(projection.blocks),
        len(projection.blocks) == 1,
        max_query_vector_budget_for_task(task_path),
        projection.input_vector_dim,
        projection.output_vector_dim,
        Float64(projection_generic_report.mean()),
        projection_single_block_mean,
        Float64(projection_report.mean()),
        Float64(scan_report.mean()),
        Float64(projection_plus_scan_report.mean()),
    )


def append_latent_proxy_projection_profile_summary_json(
    mut buffer: String, read summary: LatentProxyProjectionProfileSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"projection_block_count\":"
    buffer += String(summary.projection_block_count) + ","
    buffer += "\"single_block_fast_path_available\":"
    if summary.single_block_fast_path_available:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"query_vector_budget\":" + String(summary.query_vector_budget) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"latent_dim\":" + String(summary.latent_dim) + ","
    buffer += "\"mean_projection_generic_seconds\":"
    buffer += String(summary.mean_projection_generic_seconds) + ","
    buffer += "\"mean_projection_single_block_seconds\":"
    buffer += String(summary.mean_projection_single_block_seconds) + ","
    buffer += "\"mean_projection_seconds\":"
    buffer += String(summary.mean_projection_seconds) + ","
    buffer += "\"mean_scan_seconds\":"
    buffer += String(summary.mean_scan_seconds) + ","
    buffer += "\"mean_projection_plus_scan_seconds\":"
    buffer += String(summary.mean_projection_plus_scan_seconds)
    buffer += "}"


def latent_proxy_projection_profile_summary_json(
    read summary: LatentProxyProjectionProfileSummary
) -> String:
    var buffer = String()
    append_latent_proxy_projection_profile_summary_json(buffer, summary)
    return buffer^
