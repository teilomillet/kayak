from std.collections import List

from .ids import CollectionId, NamespaceId, SegmentId, SnapshotId, TenantId
from .validation import (
    require_non_empty_string,
    require_non_negative_int,
)


comptime SNAPSHOT_RETENTION_REASON_ACTIVE = "active_snapshot"
comptime SNAPSHOT_RETENTION_REASON_PINNED = "pinned_snapshot"
comptime SNAPSHOT_RETENTION_REASON_GENERATION = "retained_inactive_by_generation"
comptime SNAPSHOT_RETENTION_REASON_RECLAIM = "inactive_reclaim_candidate"


struct SnapshotRetentionPolicy(Copyable):
    var keep_latest_inactive_count: Int
    var pinned_snapshot_ids: List[SnapshotId]

    def __init__(out self, keep_latest_inactive_count: Int) raises:
        self = SnapshotRetentionPolicy(keep_latest_inactive_count, [])

    def __init__(
        out self,
        keep_latest_inactive_count: Int,
        read pinned_snapshot_ids: List[SnapshotId],
    ) raises:
        self.keep_latest_inactive_count = require_non_negative_int(
            keep_latest_inactive_count, "keep_latest_inactive_count"
        )
        var deduped_snapshot_ids = List[SnapshotId]()
        for snapshot_id in pinned_snapshot_ids:
            var already_present = False
            for existing in deduped_snapshot_ids:
                if existing.value == snapshot_id.value:
                    already_present = True
                    break

            if not already_present:
                deduped_snapshot_ids.append(snapshot_id.copy())

        self.pinned_snapshot_ids = deduped_snapshot_ids^


struct SnapshotRetentionDecision(Copyable):
    var snapshot_id: SnapshotId
    var generation: Int
    var segment_count: Int
    var byte_size: Int
    var retain: Bool
    var reason: String

    def __init__(
        out self,
        snapshot_id: SnapshotId,
        generation: Int,
        segment_count: Int,
        byte_size: Int,
        retain: Bool,
        reason: String,
    ) raises:
        self.snapshot_id = snapshot_id.copy()
        self.generation = require_non_negative_int(generation, "generation")
        self.segment_count = require_non_negative_int(segment_count, "segment_count")
        self.byte_size = require_non_negative_int(byte_size, "byte_size")
        self.retain = retain
        self.reason = require_non_empty_string(reason, "reason")


struct CollectionReclaimPlan(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var active_snapshot_id: String
    var total_snapshot_count: Int
    var inactive_snapshot_count: Int
    var retained_inactive_snapshot_count: Int
    var reclaimable_snapshot_count: Int
    var reclaimable_unique_segment_count: Int
    var reclaimable_unique_byte_size: Int
    var decisions: List[SnapshotRetentionDecision]
    var reclaimable_unique_segment_ids: List[SegmentId]

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        active_snapshot_id: String,
        total_snapshot_count: Int,
        inactive_snapshot_count: Int,
        retained_inactive_snapshot_count: Int,
        reclaimable_snapshot_count: Int,
        reclaimable_unique_segment_count: Int,
        reclaimable_unique_byte_size: Int,
        read decisions: List[SnapshotRetentionDecision],
        read reclaimable_unique_segment_ids: List[SegmentId],
    ) raises:
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.active_snapshot_id = active_snapshot_id.copy()
        self.total_snapshot_count = require_non_negative_int(
            total_snapshot_count, "total_snapshot_count"
        )
        self.inactive_snapshot_count = require_non_negative_int(
            inactive_snapshot_count, "inactive_snapshot_count"
        )
        self.retained_inactive_snapshot_count = require_non_negative_int(
            retained_inactive_snapshot_count, "retained_inactive_snapshot_count"
        )
        self.reclaimable_snapshot_count = require_non_negative_int(
            reclaimable_snapshot_count, "reclaimable_snapshot_count"
        )
        self.reclaimable_unique_segment_count = require_non_negative_int(
            reclaimable_unique_segment_count, "reclaimable_unique_segment_count"
        )
        self.reclaimable_unique_byte_size = require_non_negative_int(
            reclaimable_unique_byte_size, "reclaimable_unique_byte_size"
        )
        if len(decisions) != self.total_snapshot_count:
            raise Error("decisions length does not match total_snapshot_count")
        if (
            self.retained_inactive_snapshot_count + self.reclaimable_snapshot_count
            != self.inactive_snapshot_count
        ):
            raise Error(
                "inactive snapshot counts do not match retained plus reclaimable counts"
            )
        if len(reclaimable_unique_segment_ids) != self.reclaimable_unique_segment_count:
            raise Error(
                "reclaimable_unique_segment_ids length does not match reclaimable_unique_segment_count"
            )

        self.decisions = decisions.copy()
        self.reclaimable_unique_segment_ids = reclaimable_unique_segment_ids.copy()

    def has_reclaimable_segments(self) -> Bool:
        return self.reclaimable_unique_segment_count > 0
