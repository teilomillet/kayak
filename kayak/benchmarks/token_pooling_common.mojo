from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    apply_document_representation_transforms_to_documents,
    token_pooling_document_representation_transform,
)
from kayak.contracts import EncodedDocument
from kayak.eval import JudgedTask, TaskEvaluation, evaluate_query_hits, choose_primary_value
from kayak.index import PackedIndex, pack_documents
from kayak.numeric import MetricScalar, zero_metric_scalar
from kayak.runtime import ExactCpuBackend
from kayak.search import SearchHit, search_exact
from kayak.storage import StoredJudgedTask, StoredPackedIndex


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    if (root / "doc_offsets.bin").exists():
        total += file_size_bytes(root / "doc_offsets.bin")
    else:
        total += file_size_bytes(root / "doc_offsets.tsv")
    total += file_size_bytes(root / "token_vectors.bin")
    return total


def standard_token_pooling_factors(max_pool_factor: Int) -> List[Int]:
    var factors = List[Int]()

    for candidate_factor in [2, 3, 4, 6, 8]:
        var factor = candidate_factor
        if factor > max_pool_factor:
            factor = max_pool_factor
        if factor < 2:
            continue
        if len(factors) == 0 or factors[len(factors) - 1] != factor:
            factors.append(factor)

    return factors^


def supported_token_pooling_policies() -> List[String]:
    return [
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    ]


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


def build_token_pooled_documents(
    read documents: List[EncodedDocument],
    pool_factor: Int,
    pooling_policy: String,
) raises -> List[EncodedDocument]:
    return apply_document_representation_transforms_to_documents(
        documents,
        [token_pooling_document_representation_transform(pool_factor, pooling_policy)],
    )


def build_token_pooled_stored_index(
    read stored_task: StoredJudgedTask,
    pool_factor: Int,
    pooling_policy: String,
) raises -> StoredPackedIndex:
    return StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(
            build_token_pooled_documents(
                stored_task.task.documents,
                pool_factor,
                pooling_policy,
            )
        ),
    )
