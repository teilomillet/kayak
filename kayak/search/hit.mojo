from kayak.numeric import ScoreScalar


struct SearchHit(Copyable):
    var doc_id: String
    var score: ScoreScalar

    def __init__(out self, var doc_id: String, score: ScoreScalar):
        self.doc_id = doc_id^
        self.score = score
