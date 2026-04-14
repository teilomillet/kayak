from std.collections import List

from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, min_score_scalar


struct ScoredCentroidSelection(Copyable):
    var centroid_indices: List[Int]
    var centroid_scores: List[ScoreScalar]
    var baseline_correction: ScoreScalar

    def __init__(
        out self,
        var centroid_indices: List[Int],
        var centroid_scores: List[ScoreScalar],
        baseline_correction: ScoreScalar,
    ):
        self.centroid_indices = centroid_indices^
        self.centroid_scores = centroid_scores^
        self.baseline_correction = baseline_correction


struct MutableCentroidSelectionScratch:
    var token_best_scores: List[ScoreScalar]
    var token_seen_generations: List[Int]
    var token_active_doc_indices: List[Int]
    var token_active_doc_count: Int
    var generation: Int

    def __init__(out self, document_count: Int):
        self.token_best_scores = List[ScoreScalar]()
        self.token_seen_generations = List[Int]()
        self.token_active_doc_indices = List[Int]()
        self.token_active_doc_count = 0
        self.generation = 1

        for _ in range(document_count):
            self.token_best_scores.append(min_score_scalar())
            self.token_seen_generations.append(0)

    def begin_token(mut self):
        self.generation += 1
        self.token_active_doc_count = 0

    def append_token_active_doc_index(mut self, doc_index: Int):
        if self.token_active_doc_count == len(self.token_active_doc_indices):
            self.token_active_doc_indices.append(doc_index)
        else:
            self.token_active_doc_indices[self.token_active_doc_count] = doc_index
        self.token_active_doc_count += 1


def accumulate_selected_centroid_scores_with_scratch(
    read selection: ScoredCentroidSelection,
    read index: CentroidPostingIndex,
    mut scores: List[ScoreScalar],
    mut active_doc_indices: List[Int],
    mut active_flags: List[Int],
    mut scratch: MutableCentroidSelectionScratch,
    read allowed_flags: List[Int] = [],
):
    scratch.begin_token()

    for centroid_list_index in range(len(selection.centroid_indices)):
        var centroid_index = selection.centroid_indices[centroid_list_index]
        var centroid_score = selection.centroid_scores[centroid_list_index]
        var start = index.posting_offsets[centroid_index]
        var stop = index.posting_offsets[centroid_index + 1]

        for posting_index in range(start, stop):
            var doc_index = index.posting_doc_indices[posting_index]
            if len(allowed_flags) != 0 and allowed_flags[doc_index] == 0:
                continue
            var approximate_score = (
                centroid_score * ScoreScalar(index.posting_weights[posting_index])
            )

            if scratch.token_seen_generations[doc_index] != scratch.generation:
                scratch.append_token_active_doc_index(doc_index)
                scratch.token_seen_generations[doc_index] = scratch.generation
                scratch.token_best_scores[doc_index] = approximate_score
            elif approximate_score > scratch.token_best_scores[doc_index]:
                scratch.token_best_scores[doc_index] = approximate_score

    for active_index in range(scratch.token_active_doc_count):
        var doc_index = scratch.token_active_doc_indices[active_index]
        if active_flags[doc_index] == 0:
            active_doc_indices.append(doc_index)
            active_flags[doc_index] = 1

        scores[doc_index] += (
            scratch.token_best_scores[doc_index] - selection.baseline_correction
        )


def accumulate_selected_centroid_scores(
    read selection: ScoredCentroidSelection,
    read index: CentroidPostingIndex,
    mut scores: List[ScoreScalar],
    mut active_doc_indices: List[Int],
    mut active_flags: List[Int],
    read allowed_flags: List[Int] = [],
):
    var scratch = MutableCentroidSelectionScratch(len(scores))
    accumulate_selected_centroid_scores_with_scratch(
        selection,
        index,
        scores,
        active_doc_indices,
        active_flags,
        scratch,
        allowed_flags,
    )
