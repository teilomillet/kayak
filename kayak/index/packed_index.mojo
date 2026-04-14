from std.collections import List
from std.format import Writable, Writer

from kayak.numeric import VectorScalar

from .validation import require_valid_packed_index


struct PackedIndex(Copyable, Writable):
    var doc_ids: List[String]
    var doc_offsets: List[Int]
    var token_vectors: List[List[VectorScalar]]
    var vector_dim: Int
    var document_count: Int
    var total_vector_count: Int

    def __init__(
        out self,
        var doc_ids: List[String],
        var doc_offsets: List[Int],
        var token_vectors: List[List[VectorScalar]],
        vector_dim: Int,
    ) raises:
        require_valid_packed_index(
            doc_ids, doc_offsets, token_vectors, vector_dim
        )

        self.document_count = len(doc_ids)
        self.total_vector_count = len(token_vectors)
        self.doc_ids = doc_ids^
        self.doc_offsets = doc_offsets^
        self.token_vectors = token_vectors^
        self.vector_dim = vector_dim

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PackedIndex(document_count=",
            self.document_count,
            ", total_vector_count=",
            self.total_vector_count,
            ", vector_dim=",
            self.vector_dim,
            ")",
        )
