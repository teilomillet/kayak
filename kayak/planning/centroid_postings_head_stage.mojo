from std.collections import List

from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.scoring.dot import dot_product


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
    candidate_k: Int, mut scores: List[ScoreScalar],
    mut active_doc_indices: List[Int], mut active_flags: List[Int],
    read allowed_flags: List[Int] = [],
):
    var token_best_scores = List[ScoreScalar]()
    var token_active_doc_indices = List[Int]()
    var token_active_flags = List[Int]()

    for _ in range(len(scores)):
        token_best_scores.append(min_score_scalar())
        token_active_flags.append(0)

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


def centroid_posting_head_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()

    for _ in range(index.document_count):
        scores.append(zero_score_scalar())
        active_flags.append(0)

    for query_token in query_token_vectors:
        accumulate_token_best_doc_scores_head(
            query_token,
            index,
            candidate_k,
            scores,
            active_doc_indices,
            active_flags,
            allowed_flags,
        )

    return scores^
