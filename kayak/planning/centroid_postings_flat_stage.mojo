from std.collections import List

from kayak.contracts import EncodedQuery, FlatQueryDim128, build_flat_query_dim128
from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .centroid_postings_stage import insert_descending_centroid_match


comptime QUERY_TOKEN_CENTROID_FLAT_PROBE_COUNT = 2


def dot_product_flat_pair_at_generic(
    read flat_lhs: List[VectorScalar],
    lhs_offset: Int,
    read flat_rhs: List[VectorScalar],
    rhs_offset: Int,
    vector_dim: Int,
) -> ScoreScalar:
    var total = zero_score_scalar()

    for value_index in range(vector_dim):
        total += (
            flat_lhs[lhs_offset + value_index] * flat_rhs[rhs_offset + value_index]
        )

    return total


def top_centroid_indices_for_flat_query_token_generic(
    read flat_query_values: List[VectorScalar],
    query_offset: Int,
    read index: CentroidPostingIndex,
) -> List[Int]:
    var centroid_indices = List[Int]()
    var centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        insert_descending_centroid_match(
            centroid_indices,
            centroid_scores,
            centroid_index,
            dot_product_flat_pair_at_generic(
                flat_query_values,
                query_offset,
                index.flat_centroid_values,
                centroid_index * index.vector_dim,
                index.vector_dim,
            ),
            QUERY_TOKEN_CENTROID_FLAT_PROBE_COUNT,
        )

    return centroid_indices^


def top_centroid_indices_for_flat_query_token_dim128(
    read query: FlatQueryDim128, query_index: Int, read index: CentroidPostingIndex
) -> List[Int]:
    var centroid_indices = List[Int]()
    var centroid_scores = List[ScoreScalar]()
    var query_offset = query_index * COLBERT_VECTOR_DIM

    for centroid_index in range(index.centroid_count):
        insert_descending_centroid_match(
            centroid_indices,
            centroid_scores,
            centroid_index,
            dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                index.flat_centroid_values,
                centroid_index * COLBERT_VECTOR_DIM,
            ),
            QUERY_TOKEN_CENTROID_FLAT_PROBE_COUNT,
        )

    return centroid_indices^


def accumulate_token_best_doc_scores_flat_generic(
    read flat_query_values: List[VectorScalar],
    query_offset: Int,
    read index: CentroidPostingIndex,
    mut scores: List[ScoreScalar],
    mut active_doc_indices: List[Int],
    mut active_flags: List[Int],
):
    var token_best_scores = List[ScoreScalar]()
    var token_active_doc_indices = List[Int]()
    var token_active_flags = List[Int]()

    for _ in range(len(scores)):
        token_best_scores.append(min_score_scalar())
        token_active_flags.append(0)

    for centroid_index in top_centroid_indices_for_flat_query_token_generic(
        flat_query_values, query_offset, index
    ):
        var similarity = dot_product_flat_pair_at_generic(
            flat_query_values,
            query_offset,
            index.flat_centroid_values,
            centroid_index * index.vector_dim,
            index.vector_dim,
        )
        var start = index.posting_offsets[centroid_index]
        var stop = index.posting_offsets[centroid_index + 1]

        for posting_index in range(start, stop):
            var doc_index = index.posting_doc_indices[posting_index]
            var weighted_similarity = (
                similarity * ScoreScalar(index.posting_weights[posting_index])
            )

            if token_active_flags[doc_index] == 0:
                token_active_doc_indices.append(doc_index)
                token_active_flags[doc_index] = 1

            if weighted_similarity > token_best_scores[doc_index]:
                token_best_scores[doc_index] = weighted_similarity

    for doc_index in token_active_doc_indices:
        if active_flags[doc_index] == 0:
            active_doc_indices.append(doc_index)
            active_flags[doc_index] = 1

        scores[doc_index] += token_best_scores[doc_index]


def accumulate_token_best_doc_scores_flat_dim128(
    read query: FlatQueryDim128,
    query_index: Int,
    read index: CentroidPostingIndex,
    mut scores: List[ScoreScalar],
    mut active_doc_indices: List[Int],
    mut active_flags: List[Int],
):
    var token_best_scores = List[ScoreScalar]()
    var token_active_doc_indices = List[Int]()
    var token_active_flags = List[Int]()
    var query_offset = query_index * COLBERT_VECTOR_DIM

    for _ in range(len(scores)):
        token_best_scores.append(min_score_scalar())
        token_active_flags.append(0)

    for centroid_index in top_centroid_indices_for_flat_query_token_dim128(
        query, query_index, index
    ):
        var similarity = dot_product_dim128_flat_pair_at(
            query.token_values,
            query_offset,
            index.flat_centroid_values,
            centroid_index * COLBERT_VECTOR_DIM,
        )
        var start = index.posting_offsets[centroid_index]
        var stop = index.posting_offsets[centroid_index + 1]

        for posting_index in range(start, stop):
            var doc_index = index.posting_doc_indices[posting_index]
            var weighted_similarity = (
                similarity * ScoreScalar(index.posting_weights[posting_index])
            )

            if token_active_flags[doc_index] == 0:
                token_active_doc_indices.append(doc_index)
                token_active_flags[doc_index] = 1

            if weighted_similarity > token_best_scores[doc_index]:
                token_best_scores[doc_index] = weighted_similarity

    for doc_index in token_active_doc_indices:
        if active_flags[doc_index] == 0:
            active_doc_indices.append(doc_index)
            active_flags[doc_index] = 1

        scores[doc_index] += token_best_scores[doc_index]


def centroid_posting_flat_scores_for_segment_generic(
    read query: EncodedQuery, read index: CentroidPostingIndex
) -> List[ScoreScalar]:
    var flat_query_values = List[VectorScalar]()
    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()

    for token_vector in query.token_vectors:
        for value in token_vector:
            flat_query_values.append(value)

    for _ in range(index.document_count):
        scores.append(zero_score_scalar())
        active_flags.append(0)

    for query_index in range(query.vector_count):
        accumulate_token_best_doc_scores_flat_generic(
            flat_query_values,
            query_index * query.vector_dim,
            index,
            scores,
            active_doc_indices,
            active_flags,
        )

    return scores^


def centroid_posting_flat_scores_for_segment_dim128(
    read query: FlatQueryDim128, read index: CentroidPostingIndex
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()

    for _ in range(index.document_count):
        scores.append(zero_score_scalar())
        active_flags.append(0)

    for query_index in range(query.vector_count):
        accumulate_token_best_doc_scores_flat_dim128(
            query,
            query_index,
            index,
            scores,
            active_doc_indices,
            active_flags,
        )

    return scores^


def centroid_posting_flat_scores_for_segment(
    read query: EncodedQuery, read index: CentroidPostingIndex
) raises -> List[ScoreScalar]:
    if query.vector_dim != index.vector_dim:
        raise Error("centroid posting flat stage requires matching vector_dim")

    if query.vector_dim == COLBERT_VECTOR_DIM:
        return centroid_posting_flat_scores_for_segment_dim128(
            build_flat_query_dim128(query),
            index,
        )

    return centroid_posting_flat_scores_for_segment_generic(query, index)
