from std.collections import List

from kayak.contracts import EncodedQuery


struct JudgedQuery(Copyable):
    var query_id: String
    var description: String
    var query: EncodedQuery
    var relevant_doc_ids: List[String]

    def __init__(
        out self,
        var query_id: String,
        var description: String,
        var query: EncodedQuery,
        var relevant_doc_ids: List[String],
    ):
        self.query_id = query_id^
        self.description = description^
        self.query = query^
        self.relevant_doc_ids = relevant_doc_ids^
