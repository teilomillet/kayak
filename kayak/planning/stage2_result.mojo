# Final stage-2 result over one candidate window.

from std.collections import List

from .collection_hit import CollectionHit


struct Stage2Result(Copyable):
    var final_hits: List[CollectionHit]
    var segment_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int

    def __init__(
        out self,
        var final_hits: List[CollectionHit],
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ) raises:
        if segment_count < 0:
            raise Error("stage2 result segment_count must be non-negative")

        if document_count < 0:
            raise Error("stage2 result document_count must be non-negative")

        if token_count < 0:
            raise Error("stage2 result token_count must be non-negative")

        if vector_count < 0:
            raise Error("stage2 result vector_count must be non-negative")

        if byte_size < 0:
            raise Error("stage2 result byte_size must be non-negative")

        self.final_hits = final_hits^
        self.segment_count = segment_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
