from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List
from std.runtime.asyncrt import parallelism_level

from kayak.contracts import EncodedQuery, FlatQueryDim128
from kayak.index import HybridFlatDim128Index, PackedIndex
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar

from .dot128 import COLBERT_VECTOR_DIM
from .dot128_flat import dot_product_dim128_flat_at, dot_product_dim128_flat_pair_at
from .exact_scoring_config import ExactScoringConfig
from .maxsim import (
    build_vector_balanced_boundaries,
    choose_parallel_work_item_count,
    choose_parallel_work_item_count_for_query_vector_count,
)


def build_hybrid_flat_vector_balanced_boundaries(
    read index: HybridFlatDim128Index, work_item_count: Int
) -> List[Int]:
    var boundaries = List[Int]()
    var document_count = index.document_count
    boundaries.append(0)

    if work_item_count <= 1:
        boundaries.append(document_count)
        return boundaries^

    var start_doc = 0
    var total_vector_count = index.total_vector_count

    for work_item in range(work_item_count - 1):
        var remaining_work_items = work_item_count - work_item
        var remaining_vectors = total_vector_count - index.doc_offsets[start_doc]
        var target_vectors = (remaining_vectors + remaining_work_items - 1) // remaining_work_items
        var max_stop_doc = document_count - (remaining_work_items - 1)
        var stop_doc = start_doc + 1

        while (
            stop_doc < max_stop_doc
            and (index.doc_offsets[stop_doc] - index.doc_offsets[start_doc]) < target_vectors
        ):
            stop_doc += 1

        boundaries.append(stop_doc)
        start_doc = stop_doc

    boundaries.append(document_count)
    return boundaries^


def choose_parallel_work_item_count_for_hybrid_flat_index(
    query_vector_count: Int,
    read index: HybridFlatDim128Index,
    read config: ExactScoringConfig,
) -> Int:
    if query_vector_count <= 0:
        return 1

    if not config.enable_parallel_scoring:
        return 1

    if config.parallel_work_item_count_override > 0:
        var overridden_work_item_count = config.parallel_work_item_count_override
        if overridden_work_item_count > index.document_count:
            return index.document_count
        return overridden_work_item_count

    var worker_count = parallelism_level()
    if worker_count <= 1:
        return 1

    var total_similarity_pairs = query_vector_count * index.total_vector_count
    if total_similarity_pairs < 4096:
        return 1

    if not config.enable_parallel_work_item_oversubscription:
        if worker_count > index.document_count:
            return index.document_count
        return worker_count

    var max_work_items = worker_count * 4
    if max_work_items > index.document_count:
        max_work_items = index.document_count

    var work_item_count = (total_similarity_pairs + 4096 - 1) // 4096
    if work_item_count < 1:
        return 1
    if work_item_count < worker_count:
        work_item_count = worker_count
    if work_item_count > max_work_items:
        work_item_count = max_work_items

    return work_item_count


# Owns exact MaxSim scoring for the optional flat dim128 index layout.
# It relies on a nested PackedIndex only for document partitioning metadata.
def exact_score_for_hybrid_flat_document_dim128(
    read query: EncodedQuery,
    read hybrid_index: HybridFlatDim128Index,
    document_index: Int,
) -> ScoreScalar:
    var start_vector = hybrid_index.doc_offsets[document_index]
    var stop_vector = hybrid_index.doc_offsets[document_index + 1]
    var start_offset = start_vector * COLBERT_VECTOR_DIM
    var document_vector_count = stop_vector - start_vector
    var total = zero_score_scalar()

    for query_token in query.token_vectors:
        var best_similarity = min_score_scalar()

        for token_index in range(document_vector_count):
            var similarity = dot_product_dim128_flat_at(
                query_token,
                hybrid_index.token_values,
                start_offset + (token_index * COLBERT_VECTOR_DIM),
            )

            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def exact_score_for_hybrid_flat_document_dim128_with_flat_query(
    read query: FlatQueryDim128,
    read hybrid_index: HybridFlatDim128Index,
    document_index: Int,
) -> ScoreScalar:
    var start_vector = hybrid_index.doc_offsets[document_index]
    var stop_vector = hybrid_index.doc_offsets[document_index + 1]
    var start_offset = start_vector * COLBERT_VECTOR_DIM
    var document_vector_count = stop_vector - start_vector
    var total = zero_score_scalar()

    for query_index in range(query.vector_count):
        var query_offset = query_index * COLBERT_VECTOR_DIM
        var best_similarity = min_score_scalar()

        for token_index in range(document_vector_count):
            var similarity = dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                hybrid_index.token_values,
                start_offset + (token_index * COLBERT_VECTOR_DIM),
            )

            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def exact_scores_for_hybrid_flat_index_dim128(
    read query: EncodedQuery,
    read nested_index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
    read config: ExactScoringConfig,
) raises -> List[ScoreScalar]:
    if query.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 scorer requires query vector_dim=128")
    if nested_index.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 scorer requires index vector_dim=128")

    var document_count = nested_index.document_count
    var work_item_count = choose_parallel_work_item_count(
        query, nested_index, config
    )
    if work_item_count <= 1:
        var serial_scores = List[ScoreScalar]()
        for document_index in range(document_count):
            serial_scores.append(
                exact_score_for_hybrid_flat_document_dim128(
                    query, hybrid_index, document_index
                )
            )
        return serial_scores^

    var scores = List[ScoreScalar]()
    for _ in range(document_count):
        scores.append(zero_score_scalar())

    var boundaries = build_vector_balanced_boundaries(nested_index, work_item_count)
    var scores_ptr = scores.unsafe_ptr()

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = exact_score_for_hybrid_flat_document_dim128(
                query, hybrid_index, document_index
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def exact_scores_for_hybrid_flat_only_index_dim128(
    read query: EncodedQuery,
    read hybrid_index: HybridFlatDim128Index,
    read config: ExactScoringConfig,
) raises -> List[ScoreScalar]:
    if query.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 scorer requires query vector_dim=128")

    var document_count = hybrid_index.document_count
    var work_item_count = choose_parallel_work_item_count_for_hybrid_flat_index(
        query.vector_count, hybrid_index, config
    )
    if work_item_count <= 1:
        var serial_scores = List[ScoreScalar]()
        for document_index in range(document_count):
            serial_scores.append(
                exact_score_for_hybrid_flat_document_dim128(
                    query, hybrid_index, document_index
                )
            )
        return serial_scores^

    var scores = List[ScoreScalar]()
    for _ in range(document_count):
        scores.append(zero_score_scalar())

    var boundaries = build_hybrid_flat_vector_balanced_boundaries(
        hybrid_index, work_item_count
    )
    var scores_ptr = scores.unsafe_ptr()

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = exact_score_for_hybrid_flat_document_dim128(
                query, hybrid_index, document_index
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def exact_scores_for_hybrid_flat_index_dim128_with_flat_query(
    read query: FlatQueryDim128,
    read nested_index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
    read config: ExactScoringConfig,
) raises -> List[ScoreScalar]:
    if query.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 flat-query scorer requires query vector_dim=128")
    if nested_index.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 scorer requires index vector_dim=128")

    var document_count = nested_index.document_count
    var work_item_count = choose_parallel_work_item_count_for_query_vector_count(
        query.vector_count, nested_index, config
    )
    if work_item_count <= 1:
        var serial_scores = List[ScoreScalar]()
        for document_index in range(document_count):
            serial_scores.append(
                exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                    query, hybrid_index, document_index
                )
            )
        return serial_scores^

    var scores = List[ScoreScalar]()
    for _ in range(document_count):
        scores.append(zero_score_scalar())

    var boundaries = build_vector_balanced_boundaries(nested_index, work_item_count)
    var scores_ptr = scores.unsafe_ptr()

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = (
                exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                    query, hybrid_index, document_index
                )
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def exact_scores_for_hybrid_flat_only_index_dim128_with_flat_query(
    read query: FlatQueryDim128,
    read hybrid_index: HybridFlatDim128Index,
    read config: ExactScoringConfig,
) raises -> List[ScoreScalar]:
    if query.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid dim128 flat-query scorer requires query vector_dim=128")

    var document_count = hybrid_index.document_count
    var work_item_count = choose_parallel_work_item_count_for_hybrid_flat_index(
        query.vector_count, hybrid_index, config
    )
    if work_item_count <= 1:
        var serial_scores = List[ScoreScalar]()
        for document_index in range(document_count):
            serial_scores.append(
                exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                    query, hybrid_index, document_index
                )
            )
        return serial_scores^

    var scores = List[ScoreScalar]()
    for _ in range(document_count):
        scores.append(zero_score_scalar())

    var boundaries = build_hybrid_flat_vector_balanced_boundaries(
        hybrid_index, work_item_count
    )
    var scores_ptr = scores.unsafe_ptr()

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = (
                exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                    query, hybrid_index, document_index
                )
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^
