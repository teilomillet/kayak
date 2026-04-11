from std.collections import List

from kayak.numeric import VectorScalar

from .shape_validation import require_consistent_vector_dim


struct EncodedQuery(Copyable):
    var token_vectors: List[List[VectorScalar]]
    var vector_dim: Int
    var vector_count: Int

    def __init__(out self, var token_vectors: List[List[VectorScalar]]) raises:
        self.vector_dim = require_consistent_vector_dim(token_vectors, "query")
        self.vector_count = len(token_vectors)
        self.token_vectors = token_vectors^


struct EncodedDocument(Copyable):
    var doc_id: String
    var token_vectors: List[List[VectorScalar]]
    var vector_dim: Int
    var vector_count: Int

    def __init__(
        out self,
        var doc_id: String,
        var token_vectors: List[List[VectorScalar]],
    ) raises:
        self.doc_id = doc_id^
        self.vector_dim = require_consistent_vector_dim(
            token_vectors, "document"
        )
        self.vector_count = len(token_vectors)
        self.token_vectors = token_vectors^
