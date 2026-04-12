# Minimal health and metrics snapshots for the hosted service boundary.

from kayak.collections.validation import (
    require_non_empty_string,
    require_non_negative_int,
)


struct ServiceHealthStatus(Copyable):
    var status: String
    var collection_count: Int
    var live_snapshot_count: Int
    var pending_compaction_count: Int

    def __init__(
        out self,
        status: String,
        collection_count: Int,
        live_snapshot_count: Int,
        pending_compaction_count: Int,
    ) raises:
        self.status = require_non_empty_string(status, "status")
        self.collection_count = require_non_negative_int(
            collection_count, "collection_count"
        )
        self.live_snapshot_count = require_non_negative_int(
            live_snapshot_count, "live_snapshot_count"
        )
        self.pending_compaction_count = require_non_negative_int(
            pending_compaction_count, "pending_compaction_count"
        )


struct ServiceMetricsSnapshot(Copyable):
    var collection_count: Int
    var segment_count: Int
    var document_count: Int
    var vector_count: Int
    var byte_size: Int

    def __init__(
        out self,
        collection_count: Int,
        segment_count: Int,
        document_count: Int,
        vector_count: Int,
        byte_size: Int,
    ) raises:
        self.collection_count = require_non_negative_int(
            collection_count, "collection_count"
        )
        self.segment_count = require_non_negative_int(
            segment_count, "segment_count"
        )
        self.document_count = require_non_negative_int(
            document_count, "document_count"
        )
        self.vector_count = require_non_negative_int(
            vector_count, "vector_count"
        )
        self.byte_size = require_non_negative_int(byte_size, "byte_size")
