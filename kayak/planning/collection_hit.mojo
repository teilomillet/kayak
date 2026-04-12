# Search hit enriched with the sealed segment that produced it.

from kayak.numeric import ScoreScalar
from kayak.search import SearchHit


struct CollectionHit(Copyable):
    var segment_id: String
    var doc_id: String
    var score: ScoreScalar

    def __init__(
        out self, var segment_id: String, var doc_id: String, score: ScoreScalar
    ):
        self.segment_id = segment_id^
        self.doc_id = doc_id^
        self.score = score


def to_search_hit(read hit: CollectionHit) -> SearchHit:
    return SearchHit(hit.doc_id.copy(), hit.score)
