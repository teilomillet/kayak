from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List
from std.runtime.asyncrt import parallelism_level

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import (
    VECTOR_SCALAR_NAME,
    ScoreScalar,
    min_score_scalar,
    zero_score_scalar,
)
from .exact_scoring_config import ExactScoringConfig
from .dot import dot_product
from .dot128 import COLBERT_VECTOR_DIM, dot_product_dim128

comptime MIN_PARALLEL_SIMILARITY_PAIRS = 4096
comptime TARGET_CHUNKS_PER_WORKER = 4


def ceil_div(numerator: Int, denominator: Int) -> Int:
    return (numerator + denominator - 1) // denominator


def choose_parallel_work_item_count_for_query_vector_count(
    query_vector_count: Int,
    read index: PackedIndex,
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
    if total_similarity_pairs < MIN_PARALLEL_SIMILARITY_PAIRS:
        return 1

    if not config.enable_parallel_work_item_oversubscription:
        if worker_count > index.document_count:
            return index.document_count
        return worker_count

    var max_work_items = worker_count * TARGET_CHUNKS_PER_WORKER
    if max_work_items > index.document_count:
        max_work_items = index.document_count

    var work_item_count = ceil_div(
        total_similarity_pairs, MIN_PARALLEL_SIMILARITY_PAIRS
    )
    if work_item_count < 1:
        return 1
    if work_item_count < worker_count:
        work_item_count = worker_count
    if work_item_count > max_work_items:
        return max_work_items

    return work_item_count


def choose_parallel_work_item_count(
    read query: EncodedQuery,
    read index: PackedIndex,
    read config: ExactScoringConfig,
) -> Int:
    return choose_parallel_work_item_count_for_query_vector_count(
        query.vector_count, index, config
    )


def build_vector_balanced_boundaries(
    read index: PackedIndex, work_item_count: Int
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
        var remaining_vectors = (
            total_vector_count - index.doc_offsets[start_doc]
        )
        var target_vectors = ceil_div(remaining_vectors, remaining_work_items)
        var max_stop_doc = document_count - (remaining_work_items - 1)
        var stop_doc = start_doc + 1

        while (
            stop_doc < max_stop_doc
            and (
                index.doc_offsets[stop_doc] - index.doc_offsets[start_doc]
            ) < target_vectors
        ):
            stop_doc += 1

        boundaries.append(stop_doc)
        start_doc = stop_doc

    boundaries.append(document_count)
    return boundaries^


def exact_scores_for_index_serial(
    read query: EncodedQuery,
    read index: PackedIndex,
    document_count: Int,
    read config: ExactScoringConfig,
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()

    for document_index in range(document_count):
        scores.append(
            exact_score_for_document_with_config(
                query, index, document_index, config
            )
        )

    return scores^

def exact_score_for_document_generic(
    read query: EncodedQuery,
    read index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_token in query.token_vectors:
        var best_similarity = min_score_scalar()
        for token_index in range(start, stop):
            var similarity = dot_product(
                query_token, index.token_vectors[token_index]
            )

            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def exact_score_for_document_dim128(
    read query: EncodedQuery,
    read index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_token in query.token_vectors:
        var best_similarity = min_score_scalar()
        for token_index in range(start, stop):
            var similarity = dot_product_dim128(
                query_token, index.token_vectors[token_index]
            )

            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def exact_score_for_document_with_config(
    read query: EncodedQuery,
    read index: PackedIndex,
    document_index: Int,
    read config: ExactScoringConfig,
) -> ScoreScalar:
    if (
        config.enable_dim128_fast_path
        and VECTOR_SCALAR_NAME == "Float32"
        and query.vector_dim == COLBERT_VECTOR_DIM
    ):
        return exact_score_for_document_dim128(query, index, document_index)

    return exact_score_for_document_generic(query, index, document_index)


def exact_score_for_document(
    read query: EncodedQuery,
    read index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    return exact_score_for_document_with_config(
        query, index, document_index, ExactScoringConfig()
    )


def exact_scores_for_index(
    read query: EncodedQuery, read index: PackedIndex
) raises -> List[ScoreScalar]:
    return exact_scores_for_index_with_config(
        query, index, ExactScoringConfig()
    )


def exact_scores_for_index_with_config(
    read query: EncodedQuery,
    read index: PackedIndex,
    read config: ExactScoringConfig,
) raises -> List[ScoreScalar]:
    if query.vector_dim != index.vector_dim:
        raise Error("query and index must share the same vector dimension")

    var document_count = index.document_count
    var work_item_count = choose_parallel_work_item_count(query, index, config)
    if work_item_count <= 1:
        return exact_scores_for_index_serial(query, index, document_count, config)

    var scores = List[ScoreScalar]()
    for _ in range(document_count):
        scores.append(zero_score_scalar())

    var boundaries = build_vector_balanced_boundaries(index, work_item_count)
    var scores_ptr = scores.unsafe_ptr()

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = exact_score_for_document_with_config(
                query, index, document_index, config
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^
