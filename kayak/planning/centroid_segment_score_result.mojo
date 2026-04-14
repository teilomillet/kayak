# Segment-local centroid stage-1 scores.
# Owns the scorer-to-candidate-generation boundary for touched documents and the
# uniform score assigned to untouched documents. It does not own centroid
# selection or top-k insertion policy.

from std.collections import List

from kayak.numeric import ScoreScalar


struct CentroidSegmentScoreResult(Copyable):
    var active_scores: List[ScoreScalar]
    var active_doc_indices: List[Int]
    var inactive_score: ScoreScalar

    def __init__(
        out self,
        var active_scores: List[ScoreScalar],
        var active_doc_indices: List[Int],
        inactive_score: ScoreScalar,
    ):
        self.active_scores = active_scores^
        self.active_doc_indices = active_doc_indices^
        self.inactive_score = inactive_score


def materialize_centroid_segment_scores(
    read result: CentroidSegmentScoreResult, document_count: Int
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()

    for _ in range(document_count):
        scores.append(result.inactive_score)

    for active_index in range(len(result.active_doc_indices)):
        var doc_index = result.active_doc_indices[active_index]
        scores[doc_index] = result.active_scores[active_index]
    return scores^
