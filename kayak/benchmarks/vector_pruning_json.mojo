from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List
from std.pathlib import Path

from kayak.contracts import EncodedDocument
from kayak.eval import JudgedTask, TaskEvaluation, evaluate_query_hits, choose_primary_value
from kayak.index import PackedIndex, pack_documents
from kayak.numeric import MetricScalar, VectorScalar, zero_metric_scalar
from kayak.runtime import ExactCpuBackend
from kayak.search import SearchHit, search_exact
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    save_stored_packed_index,
)

from .json_common import json_escape


comptime VECTOR_PRUNING_SEARCH_MIN_SECONDS = 0.05
comptime VECTOR_PRUNING_SEARCH_MAX_SECONDS = 0.25
comptime VECTOR_PRUNING_SEARCH_MAX_ITERS = 200


struct VectorPruningSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var primary_metric: String
    var primary_value: Float64
    var document_vector_budget: Int
    var query_count: Int
    var document_count: Int
    var full_vector_count: Int
    var pruned_vector_count: Int
    var vector_dim: Int
    var mean_reference_recall_at_k: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var mean_search_seconds: Float64
    var artifact_byte_size: Int
    var artifact_bytes_per_document: Float64
    var artifact_bytes_per_vector: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var primary_metric: String,
        primary_value: Float64,
        document_vector_budget: Int,
        query_count: Int,
        document_count: Int,
        full_vector_count: Int,
        pruned_vector_count: Int,
        vector_dim: Int,
        mean_reference_recall_at_k: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        mean_search_seconds: Float64,
        artifact_byte_size: Int,
        artifact_bytes_per_document: Float64,
        artifact_bytes_per_vector: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.primary_metric = primary_metric^
        self.primary_value = primary_value
        self.document_vector_budget = document_vector_budget
        self.query_count = query_count
        self.document_count = document_count
        self.full_vector_count = full_vector_count
        self.pruned_vector_count = pruned_vector_count
        self.vector_dim = vector_dim
        self.mean_reference_recall_at_k = mean_reference_recall_at_k
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.mean_search_seconds = mean_search_seconds
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


def capped_document_vector_budget(available_count: Int, requested_budget: Int) -> Int:
    if requested_budget <= 0 or requested_budget > available_count:
        return available_count

    return requested_budget


def standard_vector_pruning_budget_sizes(max_budget: Int) -> List[Int]:
    var sizes = List[Int]()

    for candidate_size in [4, 8, 16, 32, 64, 128, max_budget]:
        var size = candidate_size
        if size > max_budget:
            size = max_budget
        if size <= 0:
            continue
        if len(sizes) == 0 or sizes[len(sizes) - 1] != size:
            sizes.append(size)

    return sizes^


def truncate_document_to_budget(
    read document: EncodedDocument, requested_budget: Int
) raises -> EncodedDocument:
    var budget = capped_document_vector_budget(
        document.vector_count, requested_budget
    )
    var token_vectors = List[List[VectorScalar]]()

    for vector_index in range(budget):
        token_vectors.append(document.token_vectors[vector_index].copy())

    return EncodedDocument(document.doc_id.copy(), token_vectors^)


def build_pruned_stored_index(
    read stored_task: StoredJudgedTask,
    document_vector_budget: Int,
) raises -> StoredPackedIndex:
    var pruned_documents = List[EncodedDocument]()
    for document in stored_task.task.documents:
        pruned_documents.append(
            truncate_document_to_budget(document, document_vector_budget)
        )

    return StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(pruned_documents),
    )


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


def evaluate_task_on_index(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
) raises -> TaskEvaluation:
    if len(task.queries) == 0:
        raise Error("cannot evaluate a task with zero queries")

    var ndcg_total = zero_metric_scalar()
    var reciprocal_rank_total = zero_metric_scalar()
    var recall_total = zero_metric_scalar()
    var success_total = zero_metric_scalar()

    for judged_query in task.queries:
        var hits = search_exact(backend, judged_query.query, index, task.k)
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


def build_vector_pruning_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read full_index: StoredPackedIndex,
    root: Path,
    document_vector_budget: Int,
) raises -> VectorPruningSummary:
    var pruned_index = build_pruned_stored_index(stored_task, document_vector_budget)
    save_stored_packed_index(root, pruned_index.copy())
    var artifact_byte_size = packed_index_storage_byte_size(root)
    var task = stored_task.task.copy()
    var evaluation = evaluate_task_on_index(backend, task, pruned_index.index)
    var reference_recall_total = 0.0
    var query_index = 0

    for judged_query in task.queries:
        var reference_hits = search_exact(
            backend, judged_query.query, full_index.index, task.k
        )
        var candidate_hits = search_exact(
            backend, judged_query.query, pruned_index.index, task.k
        )
        reference_recall_total += reference_recall_at_k(
            reference_hits, candidate_hits, task.k
        )

    def search_once() capturing raises:
        bench_compiler.keep(
            search_exact(
                backend,
                task.queries[query_index].query,
                pruned_index.index,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var search_report = run[search_once](
        num_warmup_iters=1,
        max_iters=VECTOR_PRUNING_SEARCH_MAX_ITERS,
        min_runtime_secs=VECTOR_PRUNING_SEARCH_MIN_SECONDS,
        max_runtime_secs=VECTOR_PRUNING_SEARCH_MAX_SECONDS,
    )

    return VectorPruningSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        capped_document_vector_budget(
            task.nominal_document_vector_count, document_vector_budget
        ),
        len(task.queries),
        len(task.documents),
        full_index.index.total_vector_count,
        pruned_index.index.total_vector_count,
        pruned_index.index.vector_dim,
        reference_recall_total / Float64(len(task.queries)),
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        Float64(search_report.mean()),
        artifact_byte_size,
        Float64(artifact_byte_size) / Float64(pruned_index.index.document_count),
        Float64(artifact_byte_size) / Float64(pruned_index.index.total_vector_count),
    )


def append_vector_pruning_summary_json(
    mut buffer: String, read summary: VectorPruningSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"document_vector_budget\":"
    buffer += String(summary.document_vector_budget) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"full_vector_count\":" + String(summary.full_vector_count) + ","
    buffer += "\"pruned_vector_count\":" + String(summary.pruned_vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"mean_reference_recall_at_k\":"
    buffer += String(summary.mean_reference_recall_at_k) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds) + ","
    buffer += "\"artifact_byte_size\":" + String(summary.artifact_byte_size) + ","
    buffer += "\"artifact_bytes_per_document\":"
    buffer += String(summary.artifact_bytes_per_document) + ","
    buffer += "\"artifact_bytes_per_vector\":"
    buffer += String(summary.artifact_bytes_per_vector)
    buffer += "}"


def vector_pruning_summary_json(read summary: VectorPruningSummary) -> String:
    var buffer = String()
    append_vector_pruning_summary_json(buffer, summary)
    return buffer^


def vector_pruning_summaries_json(read summaries: List[VectorPruningSummary]) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_vector_pruning_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
