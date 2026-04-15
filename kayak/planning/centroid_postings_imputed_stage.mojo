# WARP-inspired reduction over the existing centroid postings sidecar.
# This is intentionally not full WARP: Kayak does not yet store residual-coded
# cells or run WARP's native decompression path.
from std.collections import List
from std.math import sqrt

from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, VectorScalar, zero_score_scalar
from kayak.scoring.dot import dot_product

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


def finalize_imputed_centroid_selection(
    var sorted_centroid_indices: List[Int],
    var sorted_centroid_scores: List[ScoreScalar],
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ScoredCentroidSelection:
    var shortlist_count = len(sorted_centroid_indices)
    var t_prime = warp_like_t_prime(index, final_k)
    var cumulative_size = 0
    var missing_similarity_estimate = zero_score_scalar()

    for sorted_index in range(shortlist_count):
        var centroid_index = sorted_centroid_indices[sorted_index]
        cumulative_size += centroid_token_count(index, centroid_index)
        missing_similarity_estimate = sorted_centroid_scores[sorted_index]
        if cumulative_size >= t_prime:
            break

    var nprobe = effective_imputed_centroid_nprobe(shortlist_count)
    while len(sorted_centroid_indices) > nprobe:
        _ = sorted_centroid_indices.pop()
        _ = sorted_centroid_scores.pop()

    return ScoredCentroidSelection(
        sorted_centroid_indices^,
        sorted_centroid_scores^,
        missing_similarity_estimate,
    )


def small_count_imputed_centroid_selection_limit(
    read index: CentroidPostingIndex, final_k: Int, bound: Int
) -> Int:
    var nprobe = effective_imputed_centroid_nprobe(bound)
    var t_prime = warp_like_t_prime(index, final_k)
    var selection_limit = nprobe
    if t_prime > selection_limit:
        selection_limit = t_prime

    if selection_limit >= bound:
        return bound

    # finalize_imputed_centroid_selection() only needs the sorted prefix up to
    # max(nprobe, t_prime) when each centroid contributes at least one token.
    # Zero-token centroids can delay that cumulative token-count crossing, so
    # they fall back to the exact full-bound scan.
    for centroid_index in range(index.centroid_count):
        if centroid_token_count(index, centroid_index) <= 0:
            return bound

    return selection_limit


def exact_small_count_imputed_centroid_selection_limit(
    read index: CentroidPostingIndex, final_k: Int, bound: Int
) -> Int:
    if bound <= 0:
        return 0

    var nprobe = effective_imputed_centroid_nprobe(bound)
    if nprobe >= bound:
        return bound

    var t_prime = warp_like_t_prime(index, final_k)
    var sorted_token_counts = List[Int]()

    # The smallest centroid token counts form the worst-case sorted prefix for
    # the t' walk. If they cross t' after p entries then every score-ordered
    # prefix of length p crosses it as well. This tighter bound is exact, but
    # it is only worth computing when the caller can amortize it across many
    # query tokens.
    for centroid_index in range(index.centroid_count):
        var token_count = centroid_token_count(index, centroid_index)
        if token_count <= 0:
            return bound

        var insert_at = len(sorted_token_counts)
        sorted_token_counts.append(token_count)
        while insert_at > 0 and sorted_token_counts[insert_at - 1] > token_count:
            sorted_token_counts[insert_at] = sorted_token_counts[insert_at - 1]
            insert_at -= 1
        sorted_token_counts[insert_at] = token_count

    var cumulative_token_count = 0
    for prefix_index in range(len(sorted_token_counts)):
        cumulative_token_count += sorted_token_counts[prefix_index]
        var selection_limit = prefix_index + 1
        if selection_limit >= nprobe and cumulative_token_count >= t_prime:
            return selection_limit

    return bound


def centroid_selection_for_query_token(
    read query_token: List[VectorScalar],
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
            dot_product(query_token, index.centroid_vectors[centroid_index]),
            bound,
        )

    return finalize_imputed_centroid_selection(
        sorted_centroid_indices^,
        sorted_centroid_scores^,
        index,
        final_k,
    )


def centroid_posting_imputed_score_result_for_segment_with_workspace(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    final_k: Int,
    mut workspace: MutableCentroidSegmentAccumulator,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var selections = List[ScoredCentroidSelection]()
    var base_score = zero_score_scalar()

    for query_token in query_token_vectors:
        var selection = centroid_selection_for_query_token(query_token, index, final_k)
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


def centroid_posting_imputed_score_result_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> CentroidSegmentScoreResult:
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    return centroid_posting_imputed_score_result_for_segment_with_workspace(
        query_token_vectors,
        index,
        final_k,
        workspace,
        allowed_flags,
    )


def centroid_posting_imputed_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    final_k: Int,
    read allowed_flags: List[Int] = [],
) -> List[ScoreScalar]:
    return materialize_centroid_segment_scores(
        centroid_posting_imputed_score_result_for_segment(
            query_token_vectors,
            index,
            final_k,
            allowed_flags,
        ),
        index.document_count,
    )
