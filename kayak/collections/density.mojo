# Derived storage density metrics for collection and segment reports.

from kayak.numeric import MetricScalar

from .validation import require_non_negative_int


struct StorageDensity(Copyable):
    var byte_size: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var bytes_per_document: MetricScalar
    var bytes_per_token: MetricScalar
    var bytes_per_vector: MetricScalar

    def __init__(
        out self,
        byte_size: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
    ) raises:
        self.byte_size = require_non_negative_int(byte_size, "byte_size")
        self.document_count = require_non_negative_int(
            document_count, "document_count"
        )
        self.token_count = require_non_negative_int(token_count, "token_count")
        self.vector_count = require_non_negative_int(vector_count, "vector_count")

        if self.document_count == 0:
            self.bytes_per_document = MetricScalar(0.0)
        else:
            self.bytes_per_document = MetricScalar(self.byte_size) / MetricScalar(
                self.document_count
            )

        if self.token_count == 0:
            self.bytes_per_token = MetricScalar(0.0)
        else:
            self.bytes_per_token = MetricScalar(self.byte_size) / MetricScalar(
                self.token_count
            )

        if self.vector_count == 0:
            self.bytes_per_vector = MetricScalar(0.0)
        else:
            self.bytes_per_vector = MetricScalar(self.byte_size) / MetricScalar(
                self.vector_count
            )
