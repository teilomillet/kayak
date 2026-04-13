# Segment-local centroid stage-1 scores.
# Owns the scorer-to-candidate-generation boundary for touched documents and the
# uniform score assigned to untouched documents. It does not own centroid
# selection or top-k insertion policy.

from std.collections import List

from kayak.numeric import ScoreScalar


struct CentroidSegmentScoreResult(Copyable):
    var scores: List[ScoreScalar]
    var active_doc_indices: List[Int]
    var inactive_score: ScoreScalar

    def __init__(
        out self,
        var scores: List[ScoreScalar],
        var active_doc_indices: List[Int],
        inactive_score: ScoreScalar,
    ):
        self.scores = scores^
        self.active_doc_indices = active_doc_indices^
        self.inactive_score = inactive_score
