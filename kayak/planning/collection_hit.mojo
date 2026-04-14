# Search hit enriched with the sealed segment that produced it.

from kayak.numeric import ScoreScalar
from kayak.search import SearchHit


struct CollectionHit(Copyable):
    var segment_id: String
    var doc_id: String
    var score: ScoreScalar
    var segment_index: Int
    var document_index: Int

    def __init__(
        out self, var segment_id: String, var doc_id: String, score: ScoreScalar
    ):
        self.segment_id = segment_id^
        self.doc_id = doc_id^
        self.score = score
        self.segment_index = -1
        self.document_index = -1

    def __init__(
        out self,
        var segment_id: String,
        var doc_id: String,
        score: ScoreScalar,
        segment_index: Int,
        document_index: Int,
    ):
        self.segment_id = segment_id^
        self.doc_id = doc_id^
        self.score = score
        self.segment_index = segment_index
        self.document_index = document_index


def to_search_hit(read hit: CollectionHit) -> SearchHit:
    return SearchHit(hit.doc_id.copy(), hit.score)
