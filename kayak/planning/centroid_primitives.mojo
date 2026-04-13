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


def accumulate_selected_centroid_scores(
    read selection: ScoredCentroidSelection,
    read index: CentroidPostingIndex,
    mut scores: List[ScoreScalar],
    mut active_doc_indices: List[Int],
    mut active_flags: List[Int],
    read allowed_flags: List[Int] = [],
):
    var token_best_scores = List[ScoreScalar]()
    var token_active_doc_indices = List[Int]()
    var token_active_flags = List[Int]()

    for _ in range(len(scores)):
        token_best_scores.append(min_score_scalar())
        token_active_flags.append(0)

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

            if token_active_flags[doc_index] == 0:
                token_active_doc_indices.append(doc_index)
                token_active_flags[doc_index] = 1

            if approximate_score > token_best_scores[doc_index]:
                token_best_scores[doc_index] = approximate_score

    for doc_index in token_active_doc_indices:
        if active_flags[doc_index] == 0:
            active_doc_indices.append(doc_index)
            active_flags[doc_index] = 1

        scores[doc_index] += (
            token_best_scores[doc_index] - selection.baseline_correction
        )
