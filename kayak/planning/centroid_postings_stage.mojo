from std.collections import List

from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    zero_score_scalar,
)
from kayak.scoring.dot import dot_product

from .centroid_primitives import (
    MutableCentroidSelectionScratch,
    ScoredCentroidSelection,
    accumulate_selected_centroid_scores,
    accumulate_selected_centroid_scores_with_scratch,
)
from .centroid_segment_score_result import CentroidSegmentScoreResult


comptime QUERY_TOKEN_CENTROID_PROBE_COUNT = 2


def insert_descending_centroid_match(
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


def top_centroid_selection_for_query_token(
    read query_token: List[VectorScalar], read index: CentroidPostingIndex
) -> ScoredCentroidSelection:
    var centroid_indices = List[Int]()
    var centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        insert_descending_centroid_match(
            centroid_indices,
            centroid_scores,
            centroid_index,
            dot_product(query_token, index.centroid_vectors[centroid_index]),
            QUERY_TOKEN_CENTROID_PROBE_COUNT,
        )

    return ScoredCentroidSelection(centroid_indices^, centroid_scores^, ScoreScalar(0.0))


def top_centroid_indices_for_query_token(
    read query_token: List[VectorScalar], read index: CentroidPostingIndex
) -> List[Int]:
    return top_centroid_selection_for_query_token(
        query_token, index
    ).centroid_indices.copy()


def centroid_posting_score_result_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()

    for _ in range(index.document_count):
        scores.append(zero_score_scalar())
        active_flags.append(0)

    var scratch = MutableCentroidSelectionScratch(index.document_count)
    for query_token in query_token_vectors:
        accumulate_selected_centroid_scores_with_scratch(
            top_centroid_selection_for_query_token(query_token, index),
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
        zero_score_scalar(),
    )


def centroid_posting_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    read allowed_flags: List[Int] = [],
) -> List[ScoreScalar]:
    return centroid_posting_score_result_for_segment(
        query_token_vectors,
        index,
        allowed_flags,
    ).scores.copy()
