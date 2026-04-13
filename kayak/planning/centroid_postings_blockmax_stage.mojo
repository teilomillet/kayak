from std.collections import List

from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.scoring.dot import dot_product

from .centroid_segment_score_result import CentroidSegmentScoreResult
from .centroid_postings_stage import top_centroid_indices_for_query_token


struct CentroidPostingBlockmaxProfile(Copyable):
    var selected_centroid_count: Int
    var candidate_block_count: Int
    var visited_block_count: Int
    var skipped_block_count: Int
    var candidate_posting_count: Int
    var visited_posting_count: Int
    var skipped_posting_count: Int

    def __init__(
        out self,
        selected_centroid_count: Int,
        candidate_block_count: Int,
        visited_block_count: Int,
        skipped_block_count: Int,
        candidate_posting_count: Int,
        visited_posting_count: Int,
        skipped_posting_count: Int,
    ) raises:
        if selected_centroid_count < 0:
            raise Error(
                "centroid posting blockmax selected_centroid_count must be non-negative"
            )
        if candidate_block_count < 0:
            raise Error(
                "centroid posting blockmax candidate_block_count must be non-negative"
            )
        if visited_block_count < 0 or skipped_block_count < 0:
            raise Error(
                "centroid posting blockmax visited/skipped block counts must be non-negative"
            )
        if candidate_posting_count < 0:
            raise Error(
                "centroid posting blockmax candidate_posting_count must be non-negative"
            )
        if visited_posting_count < 0 or skipped_posting_count < 0:
            raise Error(
                "centroid posting blockmax visited/skipped posting counts must be non-negative"
            )
        if candidate_block_count != visited_block_count + skipped_block_count:
            raise Error(
                "centroid posting blockmax block accounting must balance"
            )
        if candidate_posting_count != visited_posting_count + skipped_posting_count:
            raise Error(
                "centroid posting blockmax posting accounting must balance"
            )

        self.selected_centroid_count = selected_centroid_count
        self.candidate_block_count = candidate_block_count
        self.visited_block_count = visited_block_count
        self.skipped_block_count = skipped_block_count
        self.candidate_posting_count = candidate_posting_count
        self.visited_posting_count = visited_posting_count
        self.skipped_posting_count = skipped_posting_count


struct CentroidPostingBlockmaxResult(Copyable):
    var score_result: CentroidSegmentScoreResult
    var profile: CentroidPostingBlockmaxProfile

    def __init__(
        out self,
        score_result: CentroidSegmentScoreResult,
        profile: CentroidPostingBlockmaxProfile,
    ):
        self.score_result = score_result.copy()
        self.profile = profile.copy()


struct MutableCentroidPostingBlockmaxProfile:
    var selected_centroid_count: Int
    var candidate_block_count: Int
    var visited_block_count: Int
    var skipped_block_count: Int
    var candidate_posting_count: Int
    var visited_posting_count: Int
    var skipped_posting_count: Int

    def __init__(out self):
        self.selected_centroid_count = 0
        self.candidate_block_count = 0
        self.visited_block_count = 0
        self.skipped_block_count = 0
        self.candidate_posting_count = 0
        self.visited_posting_count = 0
        self.skipped_posting_count = 0

    def freeze(self) raises -> CentroidPostingBlockmaxProfile:
        return CentroidPostingBlockmaxProfile(
            self.selected_centroid_count,
            self.candidate_block_count,
            self.visited_block_count,
            self.skipped_block_count,
            self.candidate_posting_count,
            self.visited_posting_count,
            self.skipped_posting_count,
        )


def bubble_up_descending_token_top_doc(
    mut top_doc_indices: List[Int],
    mut top_positions: List[Int],
    read token_best_scores: List[ScoreScalar],
    start_position: Int,
):
    var position = start_position

    while position > 0:
        var current_doc_index = top_doc_indices[position]
        var previous_doc_index = top_doc_indices[position - 1]
        if (
            token_best_scores[current_doc_index]
            <= token_best_scores[previous_doc_index]
        ):
            break

        top_doc_indices[position - 1] = current_doc_index
        top_doc_indices[position] = previous_doc_index
        top_positions[current_doc_index] = position - 1
        top_positions[previous_doc_index] = position
        position -= 1


def update_descending_token_top_docs(
    mut top_doc_indices: List[Int],
    mut top_positions: List[Int],
    read token_best_scores: List[ScoreScalar],
    doc_index: Int,
    candidate_k: Int,
):
    if candidate_k <= 0:
        return

    var existing_position = top_positions[doc_index]
    if existing_position >= 0:
        bubble_up_descending_token_top_doc(
            top_doc_indices,
            top_positions,
            token_best_scores,
            existing_position,
        )
        return

    if len(top_doc_indices) < candidate_k:
        top_doc_indices.append(doc_index)
        top_positions[doc_index] = len(top_doc_indices) - 1
        bubble_up_descending_token_top_doc(
            top_doc_indices,
            top_positions,
            token_best_scores,
            len(top_doc_indices) - 1,
        )
        return

    var threshold_doc_index = top_doc_indices[candidate_k - 1]
    if token_best_scores[doc_index] <= token_best_scores[threshold_doc_index]:
        return

    top_positions[threshold_doc_index] = -1
    top_doc_indices[candidate_k - 1] = doc_index
    top_positions[doc_index] = candidate_k - 1
    bubble_up_descending_token_top_doc(
        top_doc_indices,
        top_positions,
        token_best_scores,
        candidate_k - 1,
    )


def token_top_doc_threshold(
    read top_doc_indices: List[Int],
    read token_best_scores: List[ScoreScalar],
    candidate_k: Int,
) -> ScoreScalar:
    if candidate_k <= 0 or len(top_doc_indices) < candidate_k:
        return min_score_scalar()

    return token_best_scores[top_doc_indices[candidate_k - 1]]


def centroid_block_posting_start(
    read index: CentroidPostingIndex, centroid_index: Int, block_index: Int
) -> Int:
    return (
        index.posting_offsets[centroid_index]
        + (
            block_index - index.centroid_block_offsets[centroid_index]
        ) * index.block_size
    )


def centroid_block_posting_stop(
    read index: CentroidPostingIndex, centroid_index: Int, block_index: Int
) -> Int:
    var block_stop = centroid_block_posting_start(index, centroid_index, block_index)
    block_stop += index.block_size
    var posting_stop = index.posting_offsets[centroid_index + 1]
    if block_stop > posting_stop:
        return posting_stop

    return block_stop


def accumulate_token_best_doc_scores_blockmax(
    read query_token: List[VectorScalar], read index: CentroidPostingIndex,
    candidate_k: Int, mut scores: List[ScoreScalar],
    mut active_doc_indices: List[Int], mut active_flags: List[Int],
    mut profile: MutableCentroidPostingBlockmaxProfile,
    read allowed_flags: List[Int] = [],
):
    var token_best_scores = List[ScoreScalar]()
    var token_active_doc_indices = List[Int]()
    var token_active_flags = List[Int]()
    var token_top_doc_indices = List[Int]()
    var token_top_positions = List[Int]()

    for _ in range(len(scores)):
        token_best_scores.append(min_score_scalar())
        token_active_flags.append(0)
        token_top_positions.append(-1)

    for centroid_index in top_centroid_indices_for_query_token(query_token, index):
        profile.selected_centroid_count += 1
        var similarity = dot_product(query_token, index.centroid_vectors[centroid_index])
        var block_start_index = index.centroid_block_offsets[centroid_index]
        var block_stop_index = index.centroid_block_offsets[centroid_index + 1]

        for block_index in range(block_start_index, block_stop_index):
            var posting_start = centroid_block_posting_start(
                index, centroid_index, block_index
            )
            var posting_stop = centroid_block_posting_stop(
                index, centroid_index, block_index
            )
            var block_posting_count = posting_stop - posting_start
            profile.candidate_block_count += 1
            profile.candidate_posting_count += block_posting_count

            if (
                similarity > zero_score_scalar()
                and candidate_k > 0
                and len(token_top_doc_indices) >= candidate_k
            ):
                var threshold = token_top_doc_threshold(
                    token_top_doc_indices,
                    token_best_scores,
                    candidate_k,
                )
                var block_upper_bound = (
                    similarity * ScoreScalar(index.block_max_weights[block_index])
                )
                if block_upper_bound <= threshold:
                    profile.skipped_block_count += 1
                    profile.skipped_posting_count += block_posting_count
                    continue

            profile.visited_block_count += 1

            for posting_index in range(posting_start, posting_stop):
                profile.visited_posting_count += 1
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
                    update_descending_token_top_docs(
                        token_top_doc_indices,
                        token_top_positions,
                        token_best_scores,
                        doc_index,
                        candidate_k,
                    )

    for doc_index in token_active_doc_indices:
        if active_flags[doc_index] == 0:
            active_doc_indices.append(doc_index)
            active_flags[doc_index] = 1

        scores[doc_index] += token_best_scores[doc_index]


def centroid_posting_blockmax_scores_for_segment_profiled(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> CentroidPostingBlockmaxResult:
    var scores = List[ScoreScalar]()
    var active_flags = List[Int]()
    var active_doc_indices = List[Int]()
    var profile = MutableCentroidPostingBlockmaxProfile()

    for _ in range(index.document_count):
        scores.append(zero_score_scalar())
        active_flags.append(0)

    for query_token in query_token_vectors:
        accumulate_token_best_doc_scores_blockmax(
            query_token,
            index,
            candidate_k,
            scores,
            active_doc_indices,
            active_flags,
            profile,
            allowed_flags,
        )

    return CentroidPostingBlockmaxResult(
        CentroidSegmentScoreResult(
            scores^,
            active_doc_indices^,
            zero_score_scalar(),
        ),
        profile.freeze(),
    )


def centroid_posting_blockmax_score_result_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> CentroidSegmentScoreResult:
    return centroid_posting_blockmax_scores_for_segment_profiled(
        query_token_vectors,
        index,
        candidate_k,
        allowed_flags,
    ).score_result.copy()


def centroid_posting_blockmax_scores_for_segment(
    read query_token_vectors: List[List[VectorScalar]],
    read index: CentroidPostingIndex,
    candidate_k: Int,
    read allowed_flags: List[Int] = [],
) raises -> List[ScoreScalar]:
    return centroid_posting_blockmax_score_result_for_segment(
        query_token_vectors,
        index,
        candidate_k,
        allowed_flags,
    ).scores.copy()
