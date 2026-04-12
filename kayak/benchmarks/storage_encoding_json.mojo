from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List
from std.pathlib import Path

from kayak.eval import JudgedTask, TaskEvaluation, evaluate_query_hits, choose_primary_value
from kayak.numeric import MetricScalar, zero_metric_scalar
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    load_stored_packed_index,
    save_stored_packed_index_with_encoding,
)

from .json_common import json_escape


comptime FILE_IO_BENCH_MIN_SECONDS = 0.01
comptime FILE_IO_BENCH_MAX_SECONDS = 0.25
comptime FILE_IO_BENCH_MAX_ITERS = 20
comptime SEARCH_BENCH_MIN_SECONDS = 0.05
comptime SEARCH_BENCH_MAX_SECONDS = 0.25
comptime SEARCH_BENCH_MAX_ITERS = 200


struct StorageEncodingSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var encoding_kind: String
    var primary_metric: String
    var primary_value: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var mean_build_seconds: Float64
    var mean_load_seconds: Float64
    var mean_search_seconds: Float64
    var query_count: Int
    var document_count: Int
    var vector_count: Int
    var vector_dim: Int
    var artifact_byte_size: Int
    var artifact_bytes_per_document: Float64
    var artifact_bytes_per_vector: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var encoding_kind: String,
        var primary_metric: String,
        primary_value: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        mean_build_seconds: Float64,
        mean_load_seconds: Float64,
        mean_search_seconds: Float64,
        query_count: Int,
        document_count: Int,
        vector_count: Int,
        vector_dim: Int,
        artifact_byte_size: Int,
        artifact_bytes_per_document: Float64,
        artifact_bytes_per_vector: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.encoding_kind = encoding_kind^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.mean_build_seconds = mean_build_seconds
        self.mean_load_seconds = mean_load_seconds
        self.mean_search_seconds = mean_search_seconds
        self.query_count = query_count
        self.document_count = document_count
        self.vector_count = vector_count
        self.vector_dim = vector_dim
        self.artifact_byte_size = artifact_byte_size
        self.artifact_bytes_per_document = artifact_bytes_per_document
        self.artifact_bytes_per_vector = artifact_bytes_per_vector


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    total += file_size_bytes(root / "doc_offsets.tsv")
    total += file_size_bytes(root / "token_vectors.bin")
    return total


def require_supported_storage_encoding_kind(encoding_kind: String) raises:
    if encoding_kind == VECTOR_PAYLOAD_ENCODING_BINARY_LE:
        return

    if encoding_kind == VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE:
        return

    raise Error("unsupported storage encoding benchmark kind: " + encoding_kind)


def evaluate_task_on_index(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read stored_index: StoredPackedIndex,
) raises -> TaskEvaluation:
    if len(task.queries) == 0:
        raise Error("cannot evaluate a task with zero queries")

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

    var query_count = len(task.queries)
    var mean_ndcg_at_k = ndcg_total / MetricScalar(query_count)
    var mean_reciprocal_rank = reciprocal_rank_total / MetricScalar(query_count)
    var mean_recall_at_k = recall_total / MetricScalar(query_count)
    var success_rate_at_k = success_total / MetricScalar(query_count)

    return TaskEvaluation(
        task.family.copy(),
        task.slice_name.copy(),
        task.primary_metric.copy(),
        choose_primary_value(
            task.primary_metric,
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


def build_storage_encoding_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read stored_index: StoredPackedIndex,
    root: Path,
    encoding_kind: String,
) raises -> StorageEncodingSummary:
    require_supported_storage_encoding_kind(encoding_kind)

    def build_once() capturing raises:
        save_stored_packed_index_with_encoding(
            root,
            stored_index.copy(),
            encoding_kind,
        )

    var build_report = run[build_once](
        num_warmup_iters=1,
        max_iters=FILE_IO_BENCH_MAX_ITERS,
        min_runtime_secs=FILE_IO_BENCH_MIN_SECONDS,
        max_runtime_secs=FILE_IO_BENCH_MAX_SECONDS,
    )
    var artifact_byte_size = packed_index_storage_byte_size(root)

    def load_once() capturing raises:
        bench_compiler.keep(load_stored_packed_index(root))

    var load_report = run[load_once](
        num_warmup_iters=1,
        max_iters=FILE_IO_BENCH_MAX_ITERS,
        min_runtime_secs=FILE_IO_BENCH_MIN_SECONDS,
        max_runtime_secs=FILE_IO_BENCH_MAX_SECONDS,
    )
    var loaded_index = load_stored_packed_index(root)
    var task = stored_task.task.copy()
    var evaluation = evaluate_task_on_index(backend, task, loaded_index)
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(
            search_exact(
                backend,
                task.queries[query_index].query,
                loaded_index.index,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var search_report = run[search_once](
        num_warmup_iters=1,
        max_iters=SEARCH_BENCH_MAX_ITERS,
        min_runtime_secs=SEARCH_BENCH_MIN_SECONDS,
        max_runtime_secs=SEARCH_BENCH_MAX_SECONDS,
    )
    var document_count = loaded_index.index.document_count
    var vector_count = loaded_index.index.total_vector_count

    return StorageEncodingSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        encoding_kind.copy(),
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        Float64(build_report.mean()),
        Float64(load_report.mean()),
        Float64(search_report.mean()),
        len(task.queries),
        document_count,
        vector_count,
        loaded_index.index.vector_dim,
        artifact_byte_size,
        Float64(artifact_byte_size) / Float64(document_count),
        Float64(artifact_byte_size) / Float64(vector_count),
    )


def append_storage_encoding_summary_json(
    mut buffer: String, read summary: StorageEncodingSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"encoding_kind\":\"" + json_escape(summary.encoding_kind) + "\","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":" + String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"mean_build_seconds\":" + String(summary.mean_build_seconds) + ","
    buffer += "\"mean_load_seconds\":" + String(summary.mean_load_seconds) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"vector_count\":" + String(summary.vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"artifact_byte_size\":" + String(summary.artifact_byte_size) + ","
    buffer += "\"artifact_bytes_per_document\":"
    buffer += String(summary.artifact_bytes_per_document) + ","
    buffer += "\"artifact_bytes_per_vector\":"
    buffer += String(summary.artifact_bytes_per_vector)
    buffer += "}"


def storage_encoding_summary_json(read summary: StorageEncodingSummary) -> String:
    var buffer = String()
    append_storage_encoding_summary_json(buffer, summary)
    return buffer^


def storage_encoding_summaries_json(
    read summaries: List[StorageEncodingSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_storage_encoding_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
