from std.collections import List

from kayak.index import CentroidPostingIndex
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.scoring.dot import dot_product

from .centroid_primitives import MutableCentroidSelectionScratch
from .centroid_segment_score_result import CentroidSegmentScoreResult
from .centroid_postings_stage import top_centroid_indices_for_query_token


comptime BLOCKMAX_LINEAR_SCAN_TOP_K_LIMIT = 64


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


struct MutableCentroidPostingBlockmaxScratch:
    var selection: MutableCentroidSelectionScratch
    var token_top_positions: List[Int]
    var token_top_position_generations: List[Int]
    var use_dense_top_positions: Bool

    def __init__(out self, document_count: Int, candidate_k: Int):
        self.selection = MutableCentroidSelectionScratch(document_count)
        self.token_top_positions = List[Int]()
        self.token_top_position_generations = List[Int]()
        self.use_dense_top_positions = (
            candidate_k > BLOCKMAX_LINEAR_SCAN_TOP_K_LIMIT
        )

        if self.use_dense_top_positions:
            for _ in range(document_count):
                self.token_top_positions.append(-1)
                self.token_top_position_generations.append(0)

    def begin_token(mut self):
        self.selection.begin_token()

def dense_token_top_doc_position(
    read scratch: MutableCentroidPostingBlockmaxScratch, doc_index: Int
) -> Int:
    if (
        scratch.token_top_position_generations[doc_index]
        != scratch.selection.generation
    ):
        return -1

    return scratch.token_top_positions[doc_index]


def set_dense_token_top_doc_position(
    mut scratch: MutableCentroidPostingBlockmaxScratch,
    doc_index: Int,
    position: Int,
):
    scratch.token_top_positions[doc_index] = position
    scratch.token_top_position_generations[doc_index] = scratch.selection.generation


def clear_dense_token_top_doc_position(
    mut scratch: MutableCentroidPostingBlockmaxScratch, doc_index: Int
):
    scratch.token_top_position_generations[doc_index] = 0


def bubble_up_descending_token_top_doc(
    mut top_doc_indices: List[Int],
    mut scratch: MutableCentroidPostingBlockmaxScratch,
    start_position: Int,
    use_dense_top_positions: Bool,
):
    var position = start_position

    while position > 0:
        var current_doc_index = top_doc_indices[position]
        var previous_doc_index = top_doc_indices[position - 1]
        if (
            scratch.selection.token_best_scores[current_doc_index]
            <= scratch.selection.token_best_scores[previous_doc_index]
        ):
            break

        top_doc_indices[position - 1] = current_doc_index
        top_doc_indices[position] = previous_doc_index
        if use_dense_top_positions:
            set_dense_token_top_doc_position(
                scratch,
                current_doc_index,
                position - 1,
            )
            set_dense_token_top_doc_position(
                scratch,
                previous_doc_index,
                position,
            )
        position -= 1


def token_top_doc_position(
    read top_doc_indices: List[Int],
    read scratch: MutableCentroidPostingBlockmaxScratch,
    doc_index: Int,
    use_dense_top_positions: Bool,
) -> Int:
    if use_dense_top_positions:
        return dense_token_top_doc_position(scratch, doc_index)

    for position in range(len(top_doc_indices)):
        if top_doc_indices[position] == doc_index:
            return position

    return -1


def update_descending_token_top_docs(
    mut top_doc_indices: List[Int],
    doc_index: Int,
    candidate_k: Int,
    mut scratch: MutableCentroidPostingBlockmaxScratch,
    use_dense_top_positions: Bool,
):
    if candidate_k <= 0:
        return

    var existing_position = token_top_doc_position(
        top_doc_indices,
        scratch,
        doc_index,
        use_dense_top_positions,
    )
    if existing_position >= 0:
        bubble_up_descending_token_top_doc(
            top_doc_indices,
            scratch,
            existing_position,
            use_dense_top_positions,
        )
        return

    if len(top_doc_indices) < candidate_k:
        top_doc_indices.append(doc_index)
        if use_dense_top_positions:
            set_dense_token_top_doc_position(
                scratch,
                doc_index,
                len(top_doc_indices) - 1,
            )
        bubble_up_descending_token_top_doc(
            top_doc_indices,
            scratch,
            len(top_doc_indices) - 1,
            use_dense_top_positions,
        )
        return

    var threshold_doc_index = top_doc_indices[candidate_k - 1]
    if (
        scratch.selection.token_best_scores[doc_index]
        <= scratch.selection.token_best_scores[threshold_doc_index]
    ):
        return

    if use_dense_top_positions:
        clear_dense_token_top_doc_position(scratch, threshold_doc_index)
        set_dense_token_top_doc_position(scratch, doc_index, candidate_k - 1)
    top_doc_indices[candidate_k - 1] = doc_index
    bubble_up_descending_token_top_doc(
        top_doc_indices,
        scratch,
        candidate_k - 1,
        use_dense_top_positions,
    )


def token_top_doc_threshold(
    read top_doc_indices: List[Int],
    read scratch: MutableCentroidPostingBlockmaxScratch,
    candidate_k: Int,
) -> ScoreScalar:
    if candidate_k <= 0 or len(top_doc_indices) < candidate_k:
        return min_score_scalar()

    return scratch.selection.token_best_scores[top_doc_indices[candidate_k - 1]]


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
    mut scratch: MutableCentroidPostingBlockmaxScratch,
    mut profile: MutableCentroidPostingBlockmaxProfile,
    read allowed_flags: List[Int] = [],
):
    var token_top_doc_indices = List[Int]()
    var use_dense_top_positions = scratch.use_dense_top_positions

    scratch.begin_token()

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
                    scratch,
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

                if (
                    scratch.selection.token_seen_generations[doc_index]
                    != scratch.selection.generation
                ):
                    scratch.selection.token_active_doc_indices.append(doc_index)
                    scratch.selection.token_seen_generations[doc_index] = (
                        scratch.selection.generation
                    )
                    scratch.selection.token_best_scores[doc_index] = (
                        weighted_similarity
                    )
                    update_descending_token_top_docs(
                        token_top_doc_indices,
                        doc_index,
                        candidate_k,
                        scratch,
                        use_dense_top_positions,
                    )
                elif (
                    weighted_similarity
                    > scratch.selection.token_best_scores[doc_index]
                ):
                    scratch.selection.token_best_scores[doc_index] = (
                        weighted_similarity
                    )
                    update_descending_token_top_docs(
                        token_top_doc_indices,
                        doc_index,
                        candidate_k,
                        scratch,
                        use_dense_top_positions,
                    )

    for doc_index in scratch.selection.token_active_doc_indices:
        if active_flags[doc_index] == 0:
            active_doc_indices.append(doc_index)
            active_flags[doc_index] = 1

        scores[doc_index] += scratch.selection.token_best_scores[doc_index]


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

    var scratch = MutableCentroidPostingBlockmaxScratch(
        index.document_count,
        candidate_k,
    )
    for query_token in query_token_vectors:
        accumulate_token_best_doc_scores_blockmax(
            query_token,
            index,
            candidate_k,
            scores,
            active_doc_indices,
            active_flags,
            scratch,
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
