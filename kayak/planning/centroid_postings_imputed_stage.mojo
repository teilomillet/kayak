# WARP-inspired reduction over the existing centroid postings sidecar.
# This is intentionally not full WARP: Kayak does not yet store residual-coded
# cells or run WARP's native decompression path.
from std.collections import List
from std.math import sqrt

from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, VectorScalar, min_score_scalar, zero_score_scalar
from kayak.scoring.dot import dot_product


comptime DEFAULT_IMPUTED_CENTROID_NPROBE = 32
comptime DEFAULT_IMPUTED_CENTROID_BOUND = 128
comptime DEFAULT_T_PRIME_MAX_AT_K_100 = 50_000
comptime DEFAULT_T_PRIME_MAX_ABOVE_K_100 = 100_000


struct ImputedCentroidSelection(Copyable):
    var centroid_indices: List[Int]
    var centroid_scores: List[ScoreScalar]
    var missing_similarity_estimate: ScoreScalar

    def __init__(
        out self,
        var centroid_indices: List[Int],
        var centroid_scores: List[ScoreScalar],
        missing_similarity_estimate: ScoreScalar,
    ):
        self.centroid_indices = centroid_indices^
        self.centroid_scores = centroid_scores^
        self.missing_similarity_estimate = missing_similarity_estimate


def append_descending_centroid_index(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    centroid_index: Int,
    centroid_score: ScoreScalar,
):
    var insert_at = 0
    while (
        insert_at < len(centroid_scores)
        and centroid_scores[insert_at] >= centroid_score
    ):
        insert_at += 1

    centroid_indices.append(centroid_index)
    centroid_scores.append(centroid_score)

    var current = len(centroid_scores) - 1
    while current > insert_at:
        centroid_indices[current] = centroid_indices[current - 1]
        centroid_scores[current] = centroid_scores[current - 1]
        current -= 1

    centroid_indices[insert_at] = centroid_index
    centroid_scores[insert_at] = centroid_score


def effective_imputed_centroid_bound(centroid_count: Int) -> Int:
    if centroid_count < DEFAULT_IMPUTED_CENTROID_BOUND:
        return centroid_count

    return DEFAULT_IMPUTED_CENTROID_BOUND


def effective_imputed_centroid_nprobe(bound: Int) -> Int:
    if bound < DEFAULT_IMPUTED_CENTROID_NPROBE:
        return bound

    return DEFAULT_IMPUTED_CENTROID_NPROBE


def centroid_token_count(read index: CentroidPostingIndex, centroid_index: Int) -> Int:
    return index.centroid_token_counts[centroid_index]


def total_centroid_token_count(read index: CentroidPostingIndex) -> Int:
    return index.total_centroid_token_count


def warp_like_t_prime_max(final_k: Int) -> Int:
    if final_k > 100:
        return DEFAULT_T_PRIME_MAX_ABOVE_K_100

    return DEFAULT_T_PRIME_MAX_AT_K_100


def warp_like_t_prime(read index: CentroidPostingIndex, final_k: Int) -> Int:
    var total_token_count = total_centroid_token_count(index)
    if total_token_count <= 0:
        return 1

    # WARP's public code discretizes this heuristic in 1k-token buckets. That
    # collapses to zero on our tiny public slices, so we keep the same
    # sqrt(8n)-style scaling but clamp to at least one token.
    var proportional = Int(sqrt(Float64(8 * total_token_count)))
    if proportional <= 0:
        proportional = 1

    var t_prime_max = warp_like_t_prime_max(final_k)
    if proportional > t_prime_max:
        return t_prime_max

    return proportional


def centroid_selection_for_query_token(
    read query_token: List[VectorScalar],
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
            dot_product(query_token, index.centroid_vectors[centroid_index]),
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


def centroid_posting_imputed_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    final_k: Int,
) -> List[ScoreScalar]:
    var selections = List[ImputedCentroidSelection]()
    var base_score = zero_score_scalar()

    for query_token in query_token_vectors:
        var selection = centroid_selection_for_query_token(query_token, index, final_k)
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
