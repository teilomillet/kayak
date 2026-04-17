from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
    apply_document_representation_transforms_to_documents,
    prefix_pruning_document_representation_transform,
    token_pooling_document_representation_transform,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import StoredJudgedTask, StoredPackedIndex, save_stored_packed_index

from .json_common import json_escape
from .token_pooling_common import (
    evaluate_task_on_index,
    packed_index_storage_byte_size,
    reference_recall_at_k,
    supported_token_pooling_policies,
)
from .vector_pruning_json import standard_vector_pruning_budget_sizes


comptime TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT = "full_exact"
comptime TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING = (
    "prefix_pruning"
)
comptime TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING = "token_pooling"

comptime TRAINING_FREE_SEQUENCE_COMPRESSION_SEARCH_MIN_SECONDS = 0.05
comptime TRAINING_FREE_SEQUENCE_COMPRESSION_SEARCH_MAX_SECONDS = 0.25
comptime TRAINING_FREE_SEQUENCE_COMPRESSION_SEARCH_MAX_ITERS = 200


struct TrainingFreeSequenceCompressionSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var primary_metric: String
    var primary_value: Float64
    var method_kind: String
    var transform_policy: String
    var requested_document_vector_budget: Int
    var pool_factor: Int
    var query_count: Int
    var document_count: Int
    var full_vector_count: Int
    var transformed_vector_count: Int
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
        var method_kind: String,
        var transform_policy: String,
        requested_document_vector_budget: Int,
        pool_factor: Int,
        query_count: Int,
        document_count: Int,
        full_vector_count: Int,
        transformed_vector_count: Int,
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
        self.method_kind = method_kind^
        self.transform_policy = transform_policy^
        self.requested_document_vector_budget = requested_document_vector_budget
        self.pool_factor = pool_factor
        self.query_count = query_count
        self.document_count = document_count
        self.full_vector_count = full_vector_count
        self.transformed_vector_count = transformed_vector_count
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


def standard_training_free_sequence_compression_budget_sizes(
    max_budget: Int
) -> List[Int]:
    return standard_vector_pruning_budget_sizes(max_budget)


def pool_factor_for_target_document_vector_budget(
    available_count: Int, requested_budget: Int
) raises -> Int:
    if available_count <= 0:
        raise Error("sequence compression requires a positive available vector count")
    if requested_budget <= 0:
        raise Error("sequence compression requested budget must be positive")
    if requested_budget >= available_count:
        return 1

    var factor = available_count // requested_budget
    if available_count % requested_budget != 0:
        factor += 1
    if factor <= 0:
        return 1
    return factor


def build_transformed_stored_index_for_sequence_compression(
    read stored_task: StoredJudgedTask,
    read full_index: StoredPackedIndex,
    method_kind: String,
    requested_document_vector_budget: Int,
    transform_policy: String,
) raises -> StoredPackedIndex:
    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT:
        return full_index.copy()

    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING:
        return StoredPackedIndex(
            stored_task.dataset_id.copy(),
            stored_task.model_name.copy(),
            stored_task.vector_scalar_name.copy(),
            pack_documents(
                apply_document_representation_transforms_to_documents(
                    stored_task.task.documents,
                    [
                        prefix_pruning_document_representation_transform(
                            requested_document_vector_budget,
                            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
                        )
                    ],
                )
            ),
        )

    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING:
        var pool_factor = pool_factor_for_target_document_vector_budget(
            stored_task.task.nominal_document_vector_count,
            requested_document_vector_budget,
        )
        return StoredPackedIndex(
            stored_task.dataset_id.copy(),
            stored_task.model_name.copy(),
            stored_task.vector_scalar_name.copy(),
            pack_documents(
                apply_document_representation_transforms_to_documents(
                    stored_task.task.documents,
                    [
                        token_pooling_document_representation_transform(
                            pool_factor, transform_policy
                        )
                    ],
                )
            ),
        )

    raise Error("unsupported sequence compression method kind: " + method_kind)


def build_training_free_sequence_compression_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read full_index: StoredPackedIndex,
    root: Path,
    method_kind: String,
    requested_document_vector_budget: Int,
    transform_policy: String = "",
) raises -> TrainingFreeSequenceCompressionSummary:
    var transformed_index = build_transformed_stored_index_for_sequence_compression(
        stored_task,
        full_index,
        method_kind,
        requested_document_vector_budget,
        transform_policy,
    )

    save_stored_packed_index(root, transformed_index.copy())
    var artifact_byte_size = packed_index_storage_byte_size(root)
    var task = stored_task.task.copy()
    var evaluation = evaluate_task_on_index(backend, task, transformed_index.index)
    var reference_recall_total = 0.0
    var query_index = 0

    for judged_query in task.queries:
        var reference_hits = search_exact(
            backend, judged_query.query, full_index.index, task.k
        )
        var transformed_hits = search_exact(
            backend, judged_query.query, transformed_index.index, task.k
        )
        reference_recall_total += reference_recall_at_k(
            reference_hits, transformed_hits, task.k
        )

    def search_once() capturing raises:
        bench_compiler.keep(
            search_exact(
                backend,
                task.queries[query_index].query,
                transformed_index.index,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var search_report = run[search_once](
        num_warmup_iters=1,
        max_iters=TRAINING_FREE_SEQUENCE_COMPRESSION_SEARCH_MAX_ITERS,
        min_runtime_secs=TRAINING_FREE_SEQUENCE_COMPRESSION_SEARCH_MIN_SECONDS,
        max_runtime_secs=TRAINING_FREE_SEQUENCE_COMPRESSION_SEARCH_MAX_SECONDS,
    )

    var summary_budget = requested_document_vector_budget
    var summary_pool_factor = 0
    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT:
        summary_budget = task.nominal_document_vector_count
    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING:
        summary_pool_factor = pool_factor_for_target_document_vector_budget(
            task.nominal_document_vector_count, requested_document_vector_budget
        )

    return TrainingFreeSequenceCompressionSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        evaluation.primary_metric.copy(),
        Float64(evaluation.primary_value),
        method_kind.copy(),
        transform_policy.copy(),
        summary_budget,
        summary_pool_factor,
        len(task.queries),
        len(task.documents),
        full_index.index.total_vector_count,
        transformed_index.index.total_vector_count,
        transformed_index.index.vector_dim,
        reference_recall_total / Float64(len(task.queries)),
        Float64(evaluation.mean_ndcg_at_k),
        Float64(evaluation.mean_reciprocal_rank),
        Float64(evaluation.mean_recall_at_k),
        Float64(evaluation.success_rate_at_k),
        Float64(search_report.mean()),
        artifact_byte_size,
        Float64(artifact_byte_size) / Float64(transformed_index.index.document_count),
        Float64(artifact_byte_size)
        / Float64(transformed_index.index.total_vector_count),
    )


def append_training_free_sequence_compression_summary_json(
    mut buffer: String, read summary: TrainingFreeSequenceCompressionSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"primary_metric\":\"" + json_escape(summary.primary_metric) + "\","
    buffer += "\"primary_value\":" + String(summary.primary_value) + ","
    buffer += "\"method_kind\":\"" + json_escape(summary.method_kind) + "\","
    buffer += "\"transform_policy\":\"" + json_escape(summary.transform_policy) + "\","
    buffer += "\"requested_document_vector_budget\":"
    buffer += String(summary.requested_document_vector_budget) + ","
    buffer += "\"pool_factor\":" + String(summary.pool_factor) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"full_vector_count\":" + String(summary.full_vector_count) + ","
    buffer += "\"transformed_vector_count\":"
    buffer += String(summary.transformed_vector_count) + ","
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


def training_free_sequence_compression_summary_json(
    read summary: TrainingFreeSequenceCompressionSummary
) -> String:
    var buffer = String()
    append_training_free_sequence_compression_summary_json(buffer, summary)
    return buffer^


def training_free_sequence_compression_summaries_json(
    read summaries: List[TrainingFreeSequenceCompressionSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_training_free_sequence_compression_summary_json(
            buffer, summaries[index]
        )

    buffer += "]"
    return buffer^


def default_training_free_sequence_compression_policies() -> List[String]:
    var policies = List[String]()
    policies.append(DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX)
    for policy in supported_token_pooling_policies():
        policies.append(policy.copy())
    return policies^
