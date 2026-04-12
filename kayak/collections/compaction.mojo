# Planned segment rewrites for collection maintenance.

from std.collections import List

from .ids import CollectionId, NamespaceId, SegmentId, TenantId
from .stats import SegmentStats
from .validation import require_non_empty_string


struct CompactionPlan(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var source_segment_ids: List[SegmentId]
    var target_segment_id: SegmentId
    var reason: String
    var expected_output_stats: SegmentStats

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        source_segment_ids: List[SegmentId],
        target_segment_id: SegmentId,
        reason: String,
        expected_output_stats: SegmentStats,
    ) raises:
        if len(source_segment_ids) == 0:
            raise Error("compaction plan requires at least one source segment")

        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.source_segment_ids = source_segment_ids.copy()
        self.target_segment_id = target_segment_id.copy()
        self.reason = require_non_empty_string(reason, "compaction reason")
        self.expected_output_stats = expected_output_stats.copy()
