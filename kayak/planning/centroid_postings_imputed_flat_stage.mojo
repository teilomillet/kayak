# Flat-layout follow-on for the WARP-inspired imputed centroid stage.
# This keeps the existing imputed selection/reduction semantics while replacing
# centroid similarity computation with the flatter centroid/query layout.
from std.collections import List

from kayak.contracts import EncodedQuery, FlatQueryDim128, build_flat_query_dim128
from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, VectorScalar, zero_score_scalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .centroid_primitives import (
    MutableCentroidSegmentAccumulator,
    ScoredCentroidSelection,
    accumulate_selected_centroid_scores_with_accumulator,
)
from .imputed_centroid_shortlist import (
    insert_top_bound_centroid_match,
    sort_top_bound_centroid_matches_descending,
)
from .centroid_segment_score_result import (
    CentroidSegmentScoreResult,
    materialize_centroid_segment_scores,
)
from .centroid_postings_flat_stage import dot_product_flat_pair_at_generic
from .centroid_postings_stage import insert_descending_centroid_match
from .centroid_postings_imputed_stage import (
    effective_imputed_centroid_bound,
    finalize_imputed_centroid_selection,
    small_count_imputed_centroid_selection_limit,
)


def centroid_selection_for_flat_query_token_generic(
    read flat_query_values: List[VectorScalar],
    query_offset: Int,
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ScoredCentroidSelection:
    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var selection_limit = bound
    if index.centroid_count <= bound:
        selection_limit = small_count_imputed_centroid_selection_limit(
            index,
            final_k,
            bound,
        )
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()

    if index.centroid_count <= bound:
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
                selection_limit,
            )

        return finalize_imputed_centroid_selection(
            sorted_centroid_indices^,
            sorted_centroid_scores^,
            index,
            final_k,
        )

    for centroid_index in range(index.centroid_count):
        insert_top_bound_centroid_match(
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

    sort_top_bound_centroid_matches_descending(
        sorted_centroid_indices,
        sorted_centroid_scores,
    )

    return finalize_imputed_centroid_selection(
        sorted_centroid_indices^,
        sorted_centroid_scores^,
        index,
        final_k,
    )


def centroid_selection_for_flat_query_token_dim128(
    read query: FlatQueryDim128,
    query_index: Int,
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ScoredCentroidSelection:
    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var selection_limit = bound
    if index.centroid_count <= bound:
        selection_limit = small_count_imputed_centroid_selection_limit(
            index,
            final_k,
            bound,
        )
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()
    var query_offset = query_index * COLBERT_VECTOR_DIM

    if index.centroid_count <= bound:
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
                selection_limit,
            )

        return finalize_imputed_centroid_selection(
            sorted_centroid_indices^,
            sorted_centroid_scores^,
            index,
            final_k,
        )

    for centroid_index in range(index.centroid_count):
        insert_top_bound_centroid_match(
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

    sort_top_bound_centroid_matches_descending(
        sorted_centroid_indices,
        sorted_centroid_scores,
    )

    return finalize_imputed_centroid_selection(
        sorted_centroid_indices^,
        sorted_centroid_scores^,
        index,
        final_k,
    )


def centroid_posting_imputed_flat_score_result_for_segment_generic_with_workspace(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    mut workspace: MutableCentroidSegmentAccumulator,
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
        base_score += selection.baseline_correction
        selections.append(selection^)

    workspace.begin_segment(index.document_count)
    for selection in selections:
        accumulate_selected_centroid_scores_with_accumulator(
            selection,
            index,
            workspace,
            base_score,
            allowed_flags,
        )

    return workspace.freeze(base_score)


def centroid_posting_imputed_flat_score_result_for_segment_dim128_with_workspace(
    read query: FlatQueryDim128,
    read index: CentroidPostingIndex,
    final_k: Int,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var selections = List[ScoredCentroidSelection]()
    var base_score = zero_score_scalar()

    for query_index in range(query.vector_count):
        var selection = centroid_selection_for_flat_query_token_dim128(
            query, query_index, index, final_k
        )
        base_score += selection.baseline_correction
        selections.append(selection^)

    workspace.begin_segment(index.document_count)
    for selection in selections:
        accumulate_selected_centroid_scores_with_accumulator(
            selection,
            index,
            workspace,
            base_score,
            allowed_flags,
        )

    return workspace.freeze(base_score)


def centroid_posting_imputed_flat_score_result_for_segment_generic(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_imputed_flat_score_result_for_segment_generic_with_workspace(
        query,
        index,
        final_k,
        workspace,
        allowed_flags,
    )


def centroid_posting_imputed_flat_score_result_for_segment_dim128(
    read query: FlatQueryDim128,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_imputed_flat_score_result_for_segment_dim128_with_workspace(
        query,
        index,
        final_k,
        workspace,
        allowed_flags,
    )


def centroid_posting_imputed_flat_score_result_for_segment_with_workspace(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) raises -> CentroidSegmentScoreResult:
    if query.vector_dim != index.vector_dim:
        raise Error(
            "centroid posting imputed flat stage requires matching vector_dim"
        )

    if query.vector_dim == COLBERT_VECTOR_DIM:
        return centroid_posting_imputed_flat_score_result_for_segment_dim128_with_workspace(
            build_flat_query_dim128(query),
            index,
            final_k,
            workspace,
            allowed_flags,
        )

    return centroid_posting_imputed_flat_score_result_for_segment_generic_with_workspace(
        query,
        index,
        final_k,
        workspace,
        allowed_flags,
    )


def centroid_posting_imputed_flat_score_result_for_segment(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_imputed_flat_score_result_for_segment_with_workspace(
        query,
        index,
        final_k,
        workspace,
        allowed_flags,
    )


def centroid_posting_imputed_flat_scores_for_segment(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> List[ScoreScalar]:
    return materialize_centroid_segment_scores(
        centroid_posting_imputed_flat_score_result_for_segment(
            query,
            index,
            final_k,
            allowed_flags,
        ),
        index.document_count,
    )
