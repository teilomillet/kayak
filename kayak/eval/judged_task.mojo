from std.collections import List

from kayak.contracts import EncodedDocument

from .judged_query import JudgedQuery


struct JudgedTask(Copyable):
    var family: String
    var slice_name: String
    var why: String
    var primary_metric: String
    var k: Int
    var nominal_query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var documents: List[EncodedDocument]
    var queries: List[JudgedQuery]

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var why: String,
        var primary_metric: String,
        k: Int,
        nominal_query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        var documents: List[EncodedDocument],
        var queries: List[JudgedQuery],
    ):
        self.family = family^
        self.slice_name = slice_name^
        self.why = why^
        self.primary_metric = primary_metric^
        self.k = k
        self.nominal_query_vector_count = nominal_query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.documents = documents^
        self.queries = queries^
