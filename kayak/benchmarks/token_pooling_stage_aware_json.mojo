from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List
from std.pathlib import Path

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import evaluate_query_hits
from kayak.index import PackedIndex, pack_documents
from kayak.runtime import ExactCpuBackend
from kayak.search import SearchHit, search_exact
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    save_stored_packed_index,
)
from kayak.storage.binary_vector_codec import native_vector_scalar_byte_width

from .json_common import json_escape
from .token_pooling_common import (
    build_token_pooled_stored_index,
    packed_index_storage_byte_size,
    reference_recall_at_k,
)


struct ExactRestorationResult(Copyable):
    var hits: List[SearchHit]
    var candidate_window_vector_count: Int
    var candidate_window_byte_size: Int

    def __init__(
        out self,
        var hits: List[SearchHit],
        candidate_window_vector_count: Int,
        candidate_window_byte_size: Int,
    ):
        self.hits = hits^
        self.candidate_window_vector_count = candidate_window_vector_count
        self.candidate_window_byte_size = candidate_window_byte_size


struct TokenPoolingStageAwareSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var stage1_semantics: String
    var stage2_reference_operator: String
    var pooling_policy: String
    var pool_factor: Int
    var final_k: Int
    var candidate_k: Int
    var query_count: Int
    var document_count: Int
    var full_vector_count: Int
    var pooled_vector_count: Int
    var vector_dim: Int
    var mean_candidate_generation_seconds: Float64
    var mean_stage2_rerank_seconds: Float64
    var mean_restored_search_seconds: Float64
    var mean_candidate_hit_count: Float64
    var mean_stage1_candidate_recall_at_final_k: Float64
    var mean_restored_reference_recall_at_final_k: Float64
    var mean_stage1_ndcg_at_k: Float64
    var mean_stage1_reciprocal_rank: Float64
    var mean_stage1_recall_at_k: Float64
    var stage1_success_rate_at_k: Float64
    var mean_restored_ndcg_at_k: Float64
    var mean_restored_reciprocal_rank: Float64
    var mean_restored_recall_at_k: Float64
    var restored_success_rate_at_k: Float64
    var mean_stage2_window_vector_count: Float64
    var mean_stage2_window_byte_size: Float64
    var artifact_byte_size: Int
    var artifact_bytes_per_document: Float64
    var artifact_bytes_per_vector: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var stage1_semantics: String,
        var stage2_reference_operator: String,
        var pooling_policy: String,
        pool_factor: Int,
        final_k: Int,
        candidate_k: Int,
        query_count: Int,
        document_count: Int,
        full_vector_count: Int,
        pooled_vector_count: Int,
        vector_dim: Int,
        mean_candidate_generation_seconds: Float64,
        mean_stage2_rerank_seconds: Float64,
        mean_restored_search_seconds: Float64,
        mean_candidate_hit_count: Float64,
        mean_stage1_candidate_recall_at_final_k: Float64,
        mean_restored_reference_recall_at_final_k: Float64,
        mean_stage1_ndcg_at_k: Float64,
        mean_stage1_reciprocal_rank: Float64,
        mean_stage1_recall_at_k: Float64,
        stage1_success_rate_at_k: Float64,
        mean_restored_ndcg_at_k: Float64,
        mean_restored_reciprocal_rank: Float64,
        mean_restored_recall_at_k: Float64,
        restored_success_rate_at_k: Float64,
        mean_stage2_window_vector_count: Float64,
        mean_stage2_window_byte_size: Float64,
        artifact_byte_size: Int,
        artifact_bytes_per_document: Float64,
        artifact_bytes_per_vector: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.stage1_semantics = stage1_semantics^
        self.stage2_reference_operator = stage2_reference_operator^
        self.pooling_policy = pooling_policy^
        self.pool_factor = pool_factor
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_count = query_count
        self.document_count = document_count
        self.full_vector_count = full_vector_count
        self.pooled_vector_count = pooled_vector_count
        self.vector_dim = vector_dim
        self.mean_candidate_generation_seconds = mean_candidate_generation_seconds
        self.mean_stage2_rerank_seconds = mean_stage2_rerank_seconds
        self.mean_restored_search_seconds = mean_restored_search_seconds
        self.mean_candidate_hit_count = mean_candidate_hit_count
        self.mean_stage1_candidate_recall_at_final_k = (
            mean_stage1_candidate_recall_at_final_k
        )
        self.mean_restored_reference_recall_at_final_k = (
            mean_restored_reference_recall_at_final_k
        )
        self.mean_stage1_ndcg_at_k = mean_stage1_ndcg_at_k
        self.mean_stage1_reciprocal_rank = mean_stage1_reciprocal_rank
        self.mean_stage1_recall_at_k = mean_stage1_recall_at_k
        self.stage1_success_rate_at_k = stage1_success_rate_at_k
        self.mean_restored_ndcg_at_k = mean_restored_ndcg_at_k
        self.mean_restored_reciprocal_rank = mean_restored_reciprocal_rank
        self.mean_restored_recall_at_k = mean_restored_recall_at_k
        self.restored_success_rate_at_k = restored_success_rate_at_k
        self.mean_stage2_window_vector_count = mean_stage2_window_vector_count
        self.mean_stage2_window_byte_size = mean_stage2_window_byte_size
        self.artifact_byte_size = artifact_byte_size
        self.artifact_bytes_per_document = artifact_bytes_per_document
        self.artifact_bytes_per_vector = artifact_bytes_per_vector


def find_document_by_doc_id(
    read documents: List[EncodedDocument], doc_id: String
) raises -> EncodedDocument:
    for document in documents:
        if document.doc_id == doc_id:
            return document.copy()

    raise Error("candidate doc_id not found in stored task documents: " + doc_id)


def truncate_hits(read hits: List[SearchHit], k: Int) -> List[SearchHit]:
    var limited = List[SearchHit]()
    var limit = k
    if limit > len(hits):
        limit = len(hits)

    for index in range(limit):
        limited.append(hits[index].copy())

    return limited^


def materialize_candidate_window_index(
    read documents: List[EncodedDocument],
    read candidate_hits: List[SearchHit],
) raises -> PackedIndex:
    if len(candidate_hits) == 0:
        raise Error("cannot materialize an empty candidate window")

    var candidate_documents = List[EncodedDocument]()
    for hit in candidate_hits:
        candidate_documents.append(
            find_document_by_doc_id(documents, hit.doc_id)
        )

    return pack_documents(candidate_documents)


def exact_restore_hits_for_candidates(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read documents: List[EncodedDocument],
    read candidate_hits: List[SearchHit],
    final_k: Int,
) raises -> ExactRestorationResult:
    if len(candidate_hits) == 0:
        return ExactRestorationResult([], 0, 0)

    var candidate_index = materialize_candidate_window_index(documents, candidate_hits)
    var scalar_width = native_vector_scalar_byte_width()
    return ExactRestorationResult(
        search_exact(backend, query, candidate_index, final_k),
        candidate_index.total_vector_count,
        candidate_index.total_vector_count * candidate_index.vector_dim * scalar_width,
    )


def build_token_pooling_stage_aware_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read full_index: StoredPackedIndex,
    root: Path,
    pool_factor: Int,
    pooling_policy: String,
    candidate_k: Int,
) raises -> TokenPoolingStageAwareSummary:
    var pooled_index = build_token_pooled_stored_index(
        stored_task,
        pool_factor,
        pooling_policy,
    )

    save_stored_packed_index(root, pooled_index.copy())
    var artifact_byte_size = packed_index_storage_byte_size(root)
    var task = stored_task.task.copy()
    if len(task.queries) == 0:
        raise Error("stage-aware token-pooling benchmark requires at least one query")

    var candidate_hits_by_query = List[List[SearchHit]]()
    var candidate_hit_total = 0.0
    var stage1_candidate_recall_total = 0.0
    var restored_reference_recall_total = 0.0
    var stage1_ndcg_total = 0.0
    var stage1_reciprocal_rank_total = 0.0
    var stage1_recall_total = 0.0
    var stage1_success_total = 0.0
    var restored_ndcg_total = 0.0
    var restored_reciprocal_rank_total = 0.0
    var restored_recall_total = 0.0
    var restored_success_total = 0.0
    var stage2_window_vector_total = 0.0
    var stage2_window_byte_total = 0.0

    for judged_query in task.queries:
        var reference_hits = search_exact(
            backend,
            judged_query.query,
            full_index.index,
            task.k,
        )
        var candidate_hits = search_exact(
            backend,
            judged_query.query,
            pooled_index.index,
            candidate_k,
        )
        var restored = exact_restore_hits_for_candidates(
            backend,
            judged_query.query,
            task.documents,
            candidate_hits,
            task.k,
        )
        var stage1_final_hits = truncate_hits(candidate_hits, task.k)
        var stage1_eval = evaluate_query_hits(
            judged_query,
            stage1_final_hits,
            task.k,
            task.primary_metric,
        )
        var restored_eval = evaluate_query_hits(
            judged_query,
            restored.hits,
            task.k,
            task.primary_metric,
        )

        candidate_hits_by_query.append(candidate_hits.copy())
        candidate_hit_total += Float64(len(candidate_hits))
        stage1_candidate_recall_total += reference_recall_at_k(
            reference_hits,
            candidate_hits,
            task.k,
        )
        restored_reference_recall_total += reference_recall_at_k(
            reference_hits,
            restored.hits,
            task.k,
        )
        stage1_ndcg_total += Float64(stage1_eval.ndcg_at_k)
        stage1_reciprocal_rank_total += Float64(
            stage1_eval.reciprocal_rank_at_k
        )
        stage1_recall_total += Float64(stage1_eval.recall_at_k)
        stage1_success_total += Float64(stage1_eval.success_at_k)
        restored_ndcg_total += Float64(restored_eval.ndcg_at_k)
        restored_reciprocal_rank_total += Float64(
            restored_eval.reciprocal_rank_at_k
        )
        restored_recall_total += Float64(restored_eval.recall_at_k)
        restored_success_total += Float64(restored_eval.success_at_k)
        stage2_window_vector_total += Float64(
            restored.candidate_window_vector_count
        )
        stage2_window_byte_total += Float64(restored.candidate_window_byte_size)
    var query_index = 0

    def candidate_once() capturing raises:
        bench_compiler.keep(
            search_exact(
                backend,
                task.queries[query_index].query,
                pooled_index.index,
                candidate_k,
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

    def rerank_once() capturing raises:
        bench_compiler.keep(
            exact_restore_hits_for_candidates(
                backend,
                task.queries[query_index].query,
                task.documents,
                candidate_hits_by_query[query_index],
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var rerank_report = run[rerank_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    query_index = 0

    def restored_search_once() capturing raises:
        var candidate_hits = search_exact(
            backend,
            task.queries[query_index].query,
            pooled_index.index,
            candidate_k,
        )
        bench_compiler.keep(
            exact_restore_hits_for_candidates(
                backend,
                task.queries[query_index].query,
                task.documents,
                candidate_hits,
                task.k,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var restored_search_report = run[restored_search_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )

    var query_count = len(task.queries)
    return TokenPoolingStageAwareSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        task.family.copy(),
        task.slice_name.copy(),
        "pooled_exact_prefilter",
        "exact_late_interaction",
        pooling_policy.copy(),
        pool_factor,
        task.k,
        candidate_k,
        query_count,
        len(task.documents),
        full_index.index.total_vector_count,
        pooled_index.index.total_vector_count,
        pooled_index.index.vector_dim,
        Float64(candidate_report.mean()),
        Float64(rerank_report.mean()),
        Float64(restored_search_report.mean()),
        candidate_hit_total / Float64(query_count),
        stage1_candidate_recall_total / Float64(query_count),
        restored_reference_recall_total / Float64(query_count),
        stage1_ndcg_total / Float64(query_count),
        stage1_reciprocal_rank_total / Float64(query_count),
        stage1_recall_total / Float64(query_count),
        stage1_success_total / Float64(query_count),
        restored_ndcg_total / Float64(query_count),
        restored_reciprocal_rank_total / Float64(query_count),
        restored_recall_total / Float64(query_count),
        restored_success_total / Float64(query_count),
        stage2_window_vector_total / Float64(query_count),
        stage2_window_byte_total / Float64(query_count),
        artifact_byte_size,
        Float64(artifact_byte_size) / Float64(pooled_index.index.document_count),
        Float64(artifact_byte_size) / Float64(pooled_index.index.total_vector_count),
    )


def append_token_pooling_stage_aware_summary_json(
    mut buffer: String,
    read summary: TokenPoolingStageAwareSummary,
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"stage1_semantics\":\"" + json_escape(summary.stage1_semantics)
    buffer += "\","
    buffer += "\"stage2_reference_operator\":\""
    buffer += json_escape(summary.stage2_reference_operator) + "\","
    buffer += "\"pooling_policy\":\"" + json_escape(summary.pooling_policy) + "\","
    buffer += "\"pool_factor\":" + String(summary.pool_factor) + ","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"full_vector_count\":" + String(summary.full_vector_count) + ","
    buffer += "\"pooled_vector_count\":" + String(summary.pooled_vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"mean_candidate_generation_seconds\":"
    buffer += String(summary.mean_candidate_generation_seconds) + ","
    buffer += "\"mean_stage2_rerank_seconds\":"
    buffer += String(summary.mean_stage2_rerank_seconds) + ","
    buffer += "\"mean_restored_search_seconds\":"
    buffer += String(summary.mean_restored_search_seconds) + ","
    buffer += "\"mean_candidate_hit_count\":"
    buffer += String(summary.mean_candidate_hit_count) + ","
    buffer += "\"mean_stage1_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_stage1_candidate_recall_at_final_k) + ","
    buffer += "\"mean_restored_reference_recall_at_final_k\":"
    buffer += String(summary.mean_restored_reference_recall_at_final_k) + ","
    buffer += "\"mean_stage1_ndcg_at_k\":"
    buffer += String(summary.mean_stage1_ndcg_at_k) + ","
    buffer += "\"mean_stage1_reciprocal_rank\":"
    buffer += String(summary.mean_stage1_reciprocal_rank) + ","
    buffer += "\"mean_stage1_recall_at_k\":"
    buffer += String(summary.mean_stage1_recall_at_k) + ","
    buffer += "\"stage1_success_rate_at_k\":"
    buffer += String(summary.stage1_success_rate_at_k) + ","
    buffer += "\"mean_restored_ndcg_at_k\":"
    buffer += String(summary.mean_restored_ndcg_at_k) + ","
    buffer += "\"mean_restored_reciprocal_rank\":"
    buffer += String(summary.mean_restored_reciprocal_rank) + ","
    buffer += "\"mean_restored_recall_at_k\":"
    buffer += String(summary.mean_restored_recall_at_k) + ","
    buffer += "\"restored_success_rate_at_k\":"
    buffer += String(summary.restored_success_rate_at_k) + ","
    buffer += "\"mean_stage2_window_vector_count\":"
    buffer += String(summary.mean_stage2_window_vector_count) + ","
    buffer += "\"mean_stage2_window_byte_size\":"
    buffer += String(summary.mean_stage2_window_byte_size) + ","
    buffer += "\"artifact_byte_size\":" + String(summary.artifact_byte_size) + ","
    buffer += "\"artifact_bytes_per_document\":"
    buffer += String(summary.artifact_bytes_per_document) + ","
    buffer += "\"artifact_bytes_per_vector\":"
    buffer += String(summary.artifact_bytes_per_vector)
    buffer += "}"


def token_pooling_stage_aware_summary_json(
    read summary: TokenPoolingStageAwareSummary
) -> String:
    var buffer = String()
    append_token_pooling_stage_aware_summary_json(buffer, summary)
    return buffer^


def token_pooling_stage_aware_summaries_json(
    read summaries: List[TokenPoolingStageAwareSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_token_pooling_stage_aware_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
