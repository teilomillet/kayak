# Snapshot contract that defines one searchable collection state.

from std.collections import List

from .ids import CollectionId, NamespaceId, SegmentId, SnapshotId, TenantId
from .stats import CollectionStats
from .validation import require_non_negative_int


struct SnapshotManifest(Copyable):
    var snapshot_id: SnapshotId
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var generation: Int
    var segment_ids: List[SegmentId]
    var stats: CollectionStats

    def __init__(
        out self,
        snapshot_id: SnapshotId,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        generation: Int,
        segment_ids: List[SegmentId],
        stats: CollectionStats,
    ) raises:
        self.snapshot_id = snapshot_id.copy()
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.generation = require_non_negative_int(generation, "snapshot generation")
        self.segment_ids = segment_ids.copy()
        self.stats = stats.copy()
