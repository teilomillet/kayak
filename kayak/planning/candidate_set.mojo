# Candidate artifact emitted by the stage-1 generator.

from std.collections import List

from .collection_hit import CollectionHit


struct CandidateSet(Copyable):
    var generator_kind: String
    var hits: List[CollectionHit]
    var segment_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int

    def __init__(
        out self,
        var generator_kind: String,
        var hits: List[CollectionHit],
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ) raises:
        if segment_count < 0:
            raise Error("candidate set segment_count must be non-negative")

        if document_count < 0:
            raise Error("candidate set document_count must be non-negative")

        if token_count < 0:
            raise Error("candidate set token_count must be non-negative")

        if vector_count < 0:
            raise Error("candidate set vector_count must be non-negative")

        if byte_size < 0:
            raise Error("candidate set byte_size must be non-negative")

        self.generator_kind = generator_kind^
        self.hits = hits^
        self.segment_count = segment_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
