from std.collections import List

from kayak.contracts import EncodedQuery, FlatQueryDim128, build_flat_query_dim128
from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    zero_score_scalar,
)
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .centroid_primitives import (
    MutableCentroidSegmentAccumulator,
    ScoredCentroidSelection,
    accumulate_selected_centroid_scores_with_accumulator,
)
from .centroid_segment_score_result import (
    CentroidSegmentScoreResult,
    materialize_centroid_segment_scores,
)
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


def top_centroid_selection_for_flat_query_token_generic(
    read flat_query_values: List[VectorScalar],
    query_offset: Int,
    read index: CentroidPostingIndex,
) -> ScoredCentroidSelection:
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

    return ScoredCentroidSelection(centroid_indices^, centroid_scores^, ScoreScalar(0.0))


def top_centroid_selection_for_flat_query_token_dim128(
    read query: FlatQueryDim128, query_index: Int, read index: CentroidPostingIndex
) -> ScoredCentroidSelection:
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

    return ScoredCentroidSelection(centroid_indices^, centroid_scores^, ScoreScalar(0.0))


def centroid_posting_flat_score_result_for_segment_generic_with_workspace(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var flat_query_values = List[VectorScalar]()

    for token_vector in query.token_vectors:
        for value in token_vector:
            flat_query_values.append(value)

    workspace.begin_segment(index.document_count)
    for query_index in range(query.vector_count):
        accumulate_selected_centroid_scores_with_accumulator(
            top_centroid_selection_for_flat_query_token_generic(
                flat_query_values,
                query_index * query.vector_dim,
                index,
            ),
            index,
            workspace,
            zero_score_scalar(),
            allowed_flags,
        )

    return workspace.freeze(zero_score_scalar())


def centroid_posting_flat_score_result_for_segment_dim128_with_workspace(
    read query: FlatQueryDim128,
    read index: CentroidPostingIndex,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    workspace.begin_segment(index.document_count)
    for query_index in range(query.vector_count):
        accumulate_selected_centroid_scores_with_accumulator(
            top_centroid_selection_for_flat_query_token_dim128(
                query,
                query_index,
                index,
            ),
            index,
            workspace,
            zero_score_scalar(),
            allowed_flags,
        )

    return workspace.freeze(zero_score_scalar())


def centroid_posting_flat_score_result_for_segment_generic(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_flat_score_result_for_segment_generic_with_workspace(
        query,
        index,
        workspace,
        allowed_flags,
    )


def centroid_posting_flat_score_result_for_segment_dim128(
    read query: FlatQueryDim128,
    read index: CentroidPostingIndex,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_flat_score_result_for_segment_dim128_with_workspace(
        query,
        index,
        workspace,
        allowed_flags,
    )


def centroid_posting_flat_score_result_for_segment_with_workspace(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) raises -> CentroidSegmentScoreResult:
    if query.vector_dim != index.vector_dim:
        raise Error("centroid posting flat stage requires matching vector_dim")

    if query.vector_dim == COLBERT_VECTOR_DIM:
        return centroid_posting_flat_score_result_for_segment_dim128_with_workspace(
            build_flat_query_dim128(query),
            index,
            workspace,
            allowed_flags,
        )

    return centroid_posting_flat_score_result_for_segment_generic_with_workspace(
        query,
        index,
        workspace,
        allowed_flags,
    )


def centroid_posting_flat_score_result_for_segment(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    read allowed_flags: List[Int] = [],
) raises -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_flat_score_result_for_segment_with_workspace(
        query,
        index,
        workspace,
        allowed_flags,
    )


def centroid_posting_flat_scores_for_segment(
    read query: EncodedQuery,
    read index: CentroidPostingIndex,
    read allowed_flags: List[Int] = [],
) raises -> List[ScoreScalar]:
    return materialize_centroid_segment_scores(
        centroid_posting_flat_score_result_for_segment(
            query,
            index,
            allowed_flags,
        ),
        index.document_count,
    )
