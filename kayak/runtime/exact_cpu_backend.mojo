from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import ScoreScalar
from kayak.scoring import exact_scores_for_index


struct ExactCpuBackend(Copyable):
    def __init__(out self):
        pass

    def score_all(
        self, query: EncodedQuery, index: PackedIndex
    ) raises -> List[ScoreScalar]:
        return exact_scores_for_index(query, index)
