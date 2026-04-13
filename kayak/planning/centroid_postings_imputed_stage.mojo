# WARP-inspired reduction over the existing centroid postings sidecar.
# This is intentionally not full WARP: Kayak does not yet store residual-coded
# cells or run WARP's native decompression path.
from std.collections import List
from std.math import sqrt

from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, VectorScalar, zero_score_scalar
from kayak.scoring.dot import dot_product

from .centroid_primitives import (
    MutableCentroidSelectionScratch,
    ScoredCentroidSelection,
    accumulate_selected_centroid_scores_with_scratch,
)
from .centroid_segment_score_result import CentroidSegmentScoreResult


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


comptime DEFAULT_IMPUTED_CENTROID_NPROBE = 32
comptime DEFAULT_IMPUTED_CENTROID_BOUND = 128
comptime DEFAULT_T_PRIME_MAX_AT_K_100 = 50_000
comptime DEFAULT_T_PRIME_MAX_ABOVE_K_100 = 100_000


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
) -> ScoredCentroidSelection:
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

    return ScoredCentroidSelection(
        selected_centroid_indices^,
        selected_centroid_scores^,
        missing_similarity_estimate,
    )


def centroid_posting_imputed_score_result_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var selections = List[ScoredCentroidSelection]()
    var base_score = zero_score_scalar()

    for query_token in query_token_vectors:
        var selection = centroid_selection_for_query_token(query_token, index, final_k)
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


def centroid_posting_imputed_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> List[ScoreScalar]:
    return centroid_posting_imputed_score_result_for_segment(
        query_token_vectors,
        index,
        final_k,
        allowed_flags,
    ).scores.copy()
