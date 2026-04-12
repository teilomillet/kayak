# Aggregate counts that keep vector volume explicit at the collection boundary.

from kayak.numeric import MetricScalar

from .validation import require_non_negative_int


struct SegmentStats(Copyable):
    var document_count: Int
    var token_count: Int
    var total_vector_count: Int
    var byte_size: Int
    var average_vectors_per_document: MetricScalar

    def __init__(
        out self,
        document_count: Int,
        token_count: Int,
        total_vector_count: Int,
        byte_size: Int,
    ) raises:
        self.document_count = require_non_negative_int(
            document_count, "segment document_count"
        )
        self.token_count = require_non_negative_int(
            token_count, "segment token_count"
        )
        self.total_vector_count = require_non_negative_int(
            total_vector_count, "segment total_vector_count"
        )
        self.byte_size = require_non_negative_int(byte_size, "segment byte_size")

        if self.document_count == 0:
            if self.token_count != 0 or self.total_vector_count != 0:
                raise Error(
                    "segment counts cannot have tokens or vectors without documents"
                )

            self.average_vectors_per_document = MetricScalar(0.0)
        else:
            self.average_vectors_per_document = (
                MetricScalar(self.total_vector_count)
                / MetricScalar(self.document_count)
            )


struct CollectionStats(Copyable):
    var segment_count: Int
    var document_count: Int
    var token_count: Int
    var total_vector_count: Int
    var byte_size: Int
    var average_vectors_per_document: MetricScalar

    def __init__(
        out self,
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        total_vector_count: Int,
        byte_size: Int,
    ) raises:
        self.segment_count = require_non_negative_int(
            segment_count, "collection segment_count"
        )
        self.document_count = require_non_negative_int(
            document_count, "collection document_count"
        )
        self.token_count = require_non_negative_int(
            token_count, "collection token_count"
        )
        self.total_vector_count = require_non_negative_int(
            total_vector_count, "collection total_vector_count"
        )
        self.byte_size = require_non_negative_int(byte_size, "collection byte_size")

        if self.document_count == 0:
            if self.token_count != 0 or self.total_vector_count != 0:
                raise Error(
                    "collection counts cannot have tokens or vectors without documents"
                )

            self.average_vectors_per_document = MetricScalar(0.0)
        else:
            self.average_vectors_per_document = (
                MetricScalar(self.total_vector_count)
                / MetricScalar(self.document_count)
            )
