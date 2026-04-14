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


comptime QUERY_TOKEN_CENTROID_HEAD_PROBE_COUNT = 2
comptime DEFAULT_CENTROID_HEAD_POSTING_CAP = 16


def insert_descending_centroid_head_match(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    centroid_index: Int,
    centroid_score: ScoreScalar,
    k: Int,
):
    if k <= 0:
        return

    var insert_at = 0
    while (
        insert_at < len(centroid_scores)
        and centroid_scores[insert_at] >= centroid_score
    ):
        insert_at += 1

    if insert_at >= k:
        if len(centroid_scores) < k:
            centroid_indices.append(centroid_index)
            centroid_scores.append(centroid_score)
        return

    if len(centroid_scores) < k:
        centroid_indices.append(centroid_index)
        centroid_scores.append(centroid_score)

    var current = len(centroid_scores) - 1
    while current > insert_at:
        centroid_indices[current] = centroid_indices[current - 1]
        centroid_scores[current] = centroid_scores[current - 1]
        current -= 1

    centroid_indices[insert_at] = centroid_index
    centroid_scores[insert_at] = centroid_score


def top_centroid_indices_for_query_token_head(
    read query_token: List[VectorScalar], read index: CentroidPostingIndex
) -> List[Int]:
    var centroid_indices = List[Int]()
    var centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        insert_descending_centroid_head_match(
            centroid_indices,
            centroid_scores,
            centroid_index,
            dot_product(query_token, index.centroid_vectors[centroid_index]),
            QUERY_TOKEN_CENTROID_HEAD_PROBE_COUNT,
        )

    return centroid_indices^


def capped_centroid_head_posting_stop(
    read index: CentroidPostingIndex, centroid_index: Int, candidate_k: Int
) -> Int:
    var start = index.posting_offsets[centroid_index]
    var stop = index.posting_offsets[centroid_index + 1]
    var capped = start + candidate_k
    var fixed_cap = start + DEFAULT_CENTROID_HEAD_POSTING_CAP
    if fixed_cap < capped:
        capped = fixed_cap
    if capped < stop:
        return capped

    return stop


def accumulate_token_best_doc_scores_head(
    read query_token: List[VectorScalar], read index: CentroidPostingIndex,
    candidate_k: Int, mut accumulator: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
):
    accumulator.selection.begin_token()

    for centroid_index in top_centroid_indices_for_query_token_head(query_token, index):
        var similarity = dot_product(query_token, index.centroid_vectors[centroid_index])
        var start = index.posting_offsets[centroid_index]
        var stop = capped_centroid_head_posting_stop(index, centroid_index, candidate_k)

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


def centroid_posting_head_score_result_for_segment_with_workspace(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    workspace.begin_segment(index.document_count)
    for query_token in query_token_vectors:
        accumulate_token_best_doc_scores_head(
            query_token,
            index,
            candidate_k,
            workspace,
            allowed_flags,
        )

    return workspace.freeze(zero_score_scalar())


def centroid_posting_head_score_result_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_head_score_result_for_segment_with_workspace(
        query_token_vectors,
        index,
        candidate_k,
        workspace,
        allowed_flags,
    )


def centroid_posting_head_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) -> List[ScoreScalar]:
    return materialize_centroid_segment_scores(
        centroid_posting_head_score_result_for_segment(
            query_token_vectors,
            index,
            candidate_k,
            allowed_flags,
        ),
        index.document_count,
    )
