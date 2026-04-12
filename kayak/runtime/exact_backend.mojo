from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import ScoreScalar


# Runtime boundary for exact late-interaction scoring backends.
# CPU is the current implementation, but search/eval/planning should depend
# on this contract so a GPU-backed scorer can slot in later without changing
# higher-level orchestration.
trait ExactScoringBackend:
    def score_all(
        self, read query: EncodedQuery, read index: PackedIndex
    ) raises -> List[ScoreScalar]:
        ...
