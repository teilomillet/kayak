# Flat-layout follow-on for the WARP-inspired imputed centroid stage.
# This keeps the existing imputed selection/reduction semantics while replacing
# centroid similarity computation with the flatter centroid/query layout.
from std.collections import List

from kayak.contracts import EncodedQuery, FlatQueryDim128, build_flat_query_dim128
from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, VectorScalar, min_score_scalar, zero_score_scalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .centroid_postings_flat_stage import dot_product_flat_pair_at_generic
from .centroid_postings_imputed_stage import (
    ImputedCentroidSelection,
    append_descending_centroid_index,
    centroid_token_count,
    effective_imputed_centroid_bound,
    effective_imputed_centroid_nprobe,
    warp_like_t_prime,
)


def centroid_selection_for_flat_query_token_generic(
    read flat_query_values: List[VectorScalar],
    query_offset: Int,
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ImputedCentroidSelection:
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        append_descending_centroid_index(
            sorted_centroid_indices,
            sorted_centroid_scores,
            centroid_index,
            dot_product_flat_pair_at_generic(
                flat_query_values,
                query_offset,
                index.flat_centroid_values,
                centroid_index * index.vector_dim,
                index.vector_dim,
            ),
        )

    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var nprobe = effective_imputed_centroid_nprobe(bound)
    var selected_centroid_indices = List[Int]()
    var selected_centroid_scores = List[ScoreScalar]()

    for selection_index in range(nprobe):
        selected_centroid_indices.append(sorted_centroid_indices[selection_index])
        selected_centroid_scores.append(sorted_centroid_scores[selection_index])

    var t_prime = warp_like_t_prime(index, final_k)
    var cumulative_size = 0
    var missing_similarity_estimate = zero_score_scalar()

    for sorted_index in range(bound):
        var centroid_index = sorted_centroid_indices[sorted_index]
        cumulative_size += centroid_token_count(index, centroid_index)
        missing_similarity_estimate = sorted_centroid_scores[sorted_index]
        if cumulative_size >= t_prime:
            break

    return ImputedCentroidSelection(
        selected_centroid_indices^,
        selected_centroid_scores^,
        missing_similarity_estimate,
    )


def centroid_selection_for_flat_query_token_dim128(
    read query: FlatQueryDim128,
    query_index: Int,
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ImputedCentroidSelection:
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()
    var query_offset = query_index * COLBERT_VECTOR_DIM

    for centroid_index in range(index.centroid_count):
        append_descending_centroid_index(
            sorted_centroid_indices,
            sorted_centroid_scores,
            centroid_index,
            dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                index.flat_centroid_values,
                centroid_index * COLBERT_VECTOR_DIM,
            ),
        )

    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var nprobe = effective_imputed_centroid_nprobe(bound)
    var selected_centroid_indices = List[Int]()
    var selected_centroid_scores = List[ScoreScalar]()

    for selection_index in range(nprobe):
        selected_centroid_indices.append(sorted_centroid_indices[selection_index])
        selected_centroid_scores.append(sorted_centroid_scores[selection_index])

    var t_prime = warp_like_t_prime(index, final_k)
    var cumulative_size = 0
    var missing_similarity_estimate = zero_score_scalar()

    for sorted_index in range(bound):
        var centroid_index = sorted_centroid_indices[sorted_index]
        cumulative_size += centroid_token_count(index, centroid_index)
        missing_similarity_estimate = sorted_centroid_scores[sorted_index]
        if cumulative_size >= t_prime:
            break

    return ImputedCentroidSelection(
        selected_centroid_indices^,
        selected_centroid_scores^,
        missing_similarity_estimate,
    )


def centroid_posting_imputed_flat_scores_for_segment_generic(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
) -> List[ScoreScalar]:
    var flat_query_values = List[VectorScalar]()
    var selections = List[ImputedCentroidSelection]()
    var base_score = zero_score_scalar()

    for token_vector in query.token_vectors:
        for value in token_vector:
            flat_query_values.append(value)

    for query_index in range(query.vector_count):
        var selection = centroid_selection_for_flat_query_token_generic(
            flat_query_values,
            query_index * query.vector_dim,
            index,
            final_k,
        )
        selections.append(selection.copy())
        base_score += selection.missing_similarity_estimate

    var scores = List[ScoreScalar]()
    for _ in range(index.document_count):
        scores.append(base_score)

    for selection in selections:
        var token_best_scores = List[ScoreScalar]()
        var token_active_doc_indices = List[Int]()
        var token_active_flags = List[Int]()

        for _ in range(index.document_count):
            token_best_scores.append(min_score_scalar())
            token_active_flags.append(0)

        for centroid_list_index in range(len(selection.centroid_indices)):
            var centroid_index = selection.centroid_indices[centroid_list_index]
            var centroid_score = selection.centroid_scores[centroid_list_index]
            var start = index.posting_offsets[centroid_index]
            var stop = index.posting_offsets[centroid_index + 1]

            for posting_index in range(start, stop):
                var doc_index = index.posting_doc_indices[posting_index]
                var approximate_score = (
                    centroid_score * ScoreScalar(index.posting_weights[posting_index])
                )

                if token_active_flags[doc_index] == 0:
                    token_active_doc_indices.append(doc_index)
                    token_active_flags[doc_index] = 1

                if approximate_score > token_best_scores[doc_index]:
                    token_best_scores[doc_index] = approximate_score

        for doc_index in token_active_doc_indices:
            scores[doc_index] += (
                token_best_scores[doc_index] - selection.missing_similarity_estimate
            )

    return scores^


def centroid_posting_imputed_flat_scores_for_segment_dim128(
    read query: FlatQueryDim128,
    read index: CentroidPostingIndex,
    final_k: Int,
) -> List[ScoreScalar]:
    var selections = List[ImputedCentroidSelection]()
    var base_score = zero_score_scalar()

    for query_index in range(query.vector_count):
        var selection = centroid_selection_for_flat_query_token_dim128(
            query, query_index, index, final_k
        )
        selections.append(selection.copy())
        base_score += selection.missing_similarity_estimate

    var scores = List[ScoreScalar]()
    for _ in range(index.document_count):
        scores.append(base_score)

    for selection in selections:
        var token_best_scores = List[ScoreScalar]()
        var token_active_doc_indices = List[Int]()
        var token_active_flags = List[Int]()

        for _ in range(index.document_count):
            token_best_scores.append(min_score_scalar())
            token_active_flags.append(0)

        for centroid_list_index in range(len(selection.centroid_indices)):
            var centroid_index = selection.centroid_indices[centroid_list_index]
            var centroid_score = selection.centroid_scores[centroid_list_index]
            var start = index.posting_offsets[centroid_index]
            var stop = index.posting_offsets[centroid_index + 1]

            for posting_index in range(start, stop):
                var doc_index = index.posting_doc_indices[posting_index]
                var approximate_score = (
                    centroid_score * ScoreScalar(index.posting_weights[posting_index])
                )

                if token_active_flags[doc_index] == 0:
                    token_active_doc_indices.append(doc_index)
                    token_active_flags[doc_index] = 1

                if approximate_score > token_best_scores[doc_index]:
                    token_best_scores[doc_index] = approximate_score

        for doc_index in token_active_doc_indices:
            scores[doc_index] += (
                token_best_scores[doc_index] - selection.missing_similarity_estimate
            )

    return scores^


def centroid_posting_imputed_flat_scores_for_segment(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
) raises -> List[ScoreScalar]:
    if query.vector_dim != index.vector_dim:
        raise Error(
            "centroid posting imputed flat stage requires matching vector_dim"
        )

    if query.vector_dim == COLBERT_VECTOR_DIM:
        return centroid_posting_imputed_flat_scores_for_segment_dim128(
            build_flat_query_dim128(query),
            index,
            final_k,
        )

    return centroid_posting_imputed_flat_scores_for_segment_generic(
        query, index, final_k
    )
