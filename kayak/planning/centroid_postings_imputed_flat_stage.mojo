# Flat-layout follow-on for the WARP-inspired imputed centroid stage.
# This keeps the existing imputed selection/reduction semantics while replacing
# centroid similarity computation with the flatter centroid/query layout.
from std.collections import List

from kayak.contracts import EncodedQuery, FlatQueryDim128, build_flat_query_dim128
from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, VectorScalar, min_score_scalar, zero_score_scalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .centroid_primitives import (
    MutableCentroidSelectionScratch,
    ScoredCentroidSelection,
    accumulate_selected_centroid_scores_with_scratch,
)
from .centroid_segment_score_result import CentroidSegmentScoreResult
from .centroid_postings_flat_stage import dot_product_flat_pair_at_generic
from .centroid_postings_stage import insert_descending_centroid_match
from .centroid_postings_imputed_stage import (
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
) -> ScoredCentroidSelection:
    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        insert_descending_centroid_match(
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
            bound,
        )

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

    return ScoredCentroidSelection(
        selected_centroid_indices^,
        selected_centroid_scores^,
        missing_similarity_estimate,
    )


def centroid_selection_for_flat_query_token_dim128(
    read query: FlatQueryDim128,
    query_index: Int,
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ScoredCentroidSelection:
    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()
    var query_offset = query_index * COLBERT_VECTOR_DIM

    for centroid_index in range(index.centroid_count):
        insert_descending_centroid_match(
            sorted_centroid_indices,
            sorted_centroid_scores,
            centroid_index,
            dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                index.flat_centroid_values,
                centroid_index * COLBERT_VECTOR_DIM,
            ),
            bound,
        )

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

    return ScoredCentroidSelection(
        selected_centroid_indices^,
        selected_centroid_scores^,
        missing_similarity_estimate,
    )


def centroid_posting_imputed_flat_score_result_for_segment_generic(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var flat_query_values = List[VectorScalar]()
    var selections = List[ScoredCentroidSelection]()
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
        base_score += selection.baseline_correction

    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()
    for _ in range(index.document_count):
        scores.append(base_score)
        active_flags.append(0)

    var scratch = MutableCentroidSelectionScratch(index.document_count)
    for selection in selections:
        accumulate_selected_centroid_scores_with_scratch(
            selection,
            index,
            scores,
            active_doc_indices,
            active_flags,
            scratch,
            allowed_flags,
        )

    return CentroidSegmentScoreResult(
        scores^,
        active_doc_indices^,
        base_score,
    )


def centroid_posting_imputed_flat_score_result_for_segment_dim128(
    read query: FlatQueryDim128,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var selections = List[ScoredCentroidSelection]()
    var base_score = zero_score_scalar()

    for query_index in range(query.vector_count):
        var selection = centroid_selection_for_flat_query_token_dim128(
            query, query_index, index, final_k
        )
        selections.append(selection.copy())
        base_score += selection.baseline_correction

    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()
    for _ in range(index.document_count):
        scores.append(base_score)
        active_flags.append(0)

    var scratch = MutableCentroidSelectionScratch(index.document_count)
    for selection in selections:
        accumulate_selected_centroid_scores_with_scratch(
            selection,
            index,
            scores,
            active_doc_indices,
            active_flags,
            scratch,
            allowed_flags,
        )

    return CentroidSegmentScoreResult(
        scores^,
        active_doc_indices^,
        base_score,
    )


def centroid_posting_imputed_flat_score_result_for_segment(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> CentroidSegmentScoreResult:
    if query.vector_dim != index.vector_dim:
        raise Error(
            "centroid posting imputed flat stage requires matching vector_dim"
        )

    if query.vector_dim == COLBERT_VECTOR_DIM:
        return centroid_posting_imputed_flat_score_result_for_segment_dim128(
            build_flat_query_dim128(query),
            index,
            final_k,
            allowed_flags,
        )

    return centroid_posting_imputed_flat_score_result_for_segment_generic(
        query,
        index,
        final_k,
        allowed_flags,
    )


def centroid_posting_imputed_flat_scores_for_segment(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> List[ScoreScalar]:
    return centroid_posting_imputed_flat_score_result_for_segment(
        query,
        index,
        final_k,
        allowed_flags,
    ).scores.copy()
