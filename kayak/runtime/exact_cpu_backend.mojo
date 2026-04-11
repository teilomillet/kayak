from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import ScoreScalar
from kayak.scoring import ExactScoringConfig, exact_scores_for_index_with_config


struct ExactCpuBackend(Copyable):
    var scoring_config: ExactScoringConfig

    def __init__(out self):
        self.scoring_config = ExactScoringConfig()

    def __init__(out self, var scoring_config: ExactScoringConfig):
        self.scoring_config = scoring_config^

    def score_all(
        self, query: EncodedQuery, index: PackedIndex
    ) raises -> List[ScoreScalar]:
        return exact_scores_for_index_with_config(
            query, index, self.scoring_config
        )
