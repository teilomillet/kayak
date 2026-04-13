from std.collections import List

from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.scoring.dot import dot_product

from .centroid_postings_head_stage import top_centroid_indices_for_query_token_head


comptime DEFAULT_AUTO_CENTROID_HEAD_POSTING_CAP = 16
comptime EXPANDED_AUTO_CENTROID_HEAD_POSTING_CAP = 32


def auto_centroid_head_base_posting_cap(
    read index: CentroidPostingIndex, candidate_k: Int, query_vector_count: Int
) -> Int:
    var posting_cap = DEFAULT_AUTO_CENTROID_HEAD_POSTING_CAP

    # The public posting-cap sweep showed that low query-vector budgets often
    # need a wider head window, especially when the centroid budget is either
    # very large. The fixed-cap `16` baseline was already strong, so the safe
    # policy is expansion-only and only on the high-centroid-budget regime.
    if query_vector_count <= 8:
        if index.centroid_count >= 96:
            posting_cap = candidate_k
    elif query_vector_count <= 16:
        if index.centroid_count >= 96:
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
        index, candidate_k, query_vector_count
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
    candidate_k: Int, query_vector_count: Int, mut scores: List[ScoreScalar],
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


def centroid_posting_head_auto_scores_for_segment(
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
        accumulate_token_best_doc_scores_head_auto(
            query_token,
            index,
            candidate_k,
            len(query_token_vectors),
            scores,
            active_doc_indices,
            active_flags,
            allowed_flags,
        )

    return scores^
