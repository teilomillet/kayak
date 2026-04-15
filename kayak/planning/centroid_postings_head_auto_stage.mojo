from std.collections import List

from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    zero_score_scalar,
)
from kayak.scoring.dot import dot_product

from .centroid_primitives import MutableCentroidSegmentAccumulator
from .centroid_segment_score_result import (
    CentroidSegmentScoreResult,
    materialize_centroid_segment_scores,
)
from .centroid_postings_head_stage import top_centroid_indices_for_query_token_head


comptime DEFAULT_AUTO_CENTROID_HEAD_POSTING_CAP = 16
comptime EXPANDED_AUTO_CENTROID_HEAD_POSTING_CAP = 32
comptime SHRUNK_AUTO_CENTROID_HEAD_POSTING_CAP = 4
comptime MID_WINDOW_AUTO_CENTROID_HEAD_POSTING_CAP = 8


def auto_centroid_head_base_posting_cap(
    candidate_k: Int, query_vector_count: Int, centroid_count: Int
) -> Int:
    var posting_cap = DEFAULT_AUTO_CENTROID_HEAD_POSTING_CAP

    # Long-query hard-recall slices were measured to benefit from a tighter
    # per-centroid head when the shortlist itself is already tiny. Keep that
    # shrink narrowly scoped until a broader public sweep justifies more.
    if query_vector_count >= 24:
        if candidate_k <= 10:
            posting_cap = SHRUNK_AUTO_CENTROID_HEAD_POSTING_CAP
        elif candidate_k <= 20:
            posting_cap = MID_WINDOW_AUTO_CENTROID_HEAD_POSTING_CAP
        elif candidate_k <= 40:
            posting_cap = SHRUNK_AUTO_CENTROID_HEAD_POSTING_CAP

    # The public posting-cap sweep showed that low query-vector budgets often
    # need a wider head window, especially when the centroid budget is either
    # very large. The fixed-cap `16` baseline was already strong, so the safe
    # policy is expansion-only and only on the high-centroid-budget regime.
    if query_vector_count <= 8:
        if centroid_count >= 96:
            posting_cap = candidate_k
    elif query_vector_count <= 16:
        if centroid_count >= 96:
            posting_cap = EXPANDED_AUTO_CENTROID_HEAD_POSTING_CAP

    if posting_cap > candidate_k:
        return candidate_k

    return posting_cap


def auto_centroid_head_posting_cap_for_centroid(
    read index: CentroidPostingIndex,
    centroid_index: Int,
    candidate_k: Int,
    query_vector_count: Int,
) -> Int:
    var posting_count = index.centroid_document_counts[centroid_index]
    if posting_count <= 0:
        return 0

    var posting_cap = auto_centroid_head_base_posting_cap(
        candidate_k, query_vector_count, index.centroid_count
    )

    # Keep full access to rare centroids because they are already cheap and are
    # more likely to carry specific evidence.
    if posting_count <= posting_cap:
        return posting_count

    return posting_cap


def auto_centroid_head_posting_stop(
    read index: CentroidPostingIndex,
    centroid_index: Int,
    candidate_k: Int,
    query_vector_count: Int,
) -> Int:
    var start = index.posting_offsets[centroid_index]
    var stop = index.posting_offsets[centroid_index + 1]
    var posting_cap = auto_centroid_head_posting_cap_for_centroid(
        index, centroid_index, candidate_k, query_vector_count
    )
    if posting_cap <= 0:
        return start

    var capped = start + posting_cap
    if capped < stop:
        return capped

    return stop


def accumulate_token_best_doc_scores_head_auto(
    read query_token: List[VectorScalar], read index: CentroidPostingIndex,
    candidate_k: Int, query_vector_count: Int,
    mut accumulator: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
):
    accumulator.selection.begin_token()

    for centroid_index in top_centroid_indices_for_query_token_head(query_token, index):
        var similarity = dot_product(query_token, index.centroid_vectors[centroid_index])
        var start = index.posting_offsets[centroid_index]
        var stop = auto_centroid_head_posting_stop(
            index,
            centroid_index,
            candidate_k,
            query_vector_count,
        )

        for posting_index in range(start, stop):
            var doc_index = index.posting_doc_indices[posting_index]
            if len(allowed_flags) != 0 and allowed_flags[doc_index] == 0:
                continue
            var weighted_similarity = (
                similarity * ScoreScalar(index.posting_weights[posting_index])
            )

            if (
                accumulator.selection.token_seen_generations[doc_index]
                != accumulator.selection.generation
            ):
                accumulator.selection.append_token_active_doc_index(doc_index)
                accumulator.selection.token_seen_generations[doc_index] = (
                    accumulator.selection.generation
                )
                accumulator.selection.token_best_scores[doc_index] = (
                    weighted_similarity
                )
            elif (
                weighted_similarity
                > accumulator.selection.token_best_scores[doc_index]
            ):
                accumulator.selection.token_best_scores[doc_index] = (
                    weighted_similarity
                )

    for active_index in range(accumulator.selection.token_active_doc_count):
        var doc_index = accumulator.selection.token_active_doc_indices[active_index]
        accumulator.record_score_delta(
            doc_index,
            zero_score_scalar(),
            accumulator.selection.token_best_scores[doc_index],
        )


def centroid_posting_head_auto_score_result_for_segment_with_workspace(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    workspace.begin_segment(index.document_count)
    for query_token in query_token_vectors:
        accumulate_token_best_doc_scores_head_auto(
            query_token,
            index,
            candidate_k,
            len(query_token_vectors),
            workspace,
            allowed_flags,
        )

    return workspace.freeze(zero_score_scalar())


def centroid_posting_head_auto_score_result_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_head_auto_score_result_for_segment_with_workspace(
        query_token_vectors,
        index,
        candidate_k,
        workspace,
        allowed_flags,
    )


def centroid_posting_head_auto_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) -> List[ScoreScalar]:
    return materialize_centroid_segment_scores(
        centroid_posting_head_auto_score_result_for_segment(
            query_token_vectors,
            index,
            candidate_k,
            allowed_flags,
        ),
        index.document_count,
    )
