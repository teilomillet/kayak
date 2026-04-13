from std.collections import List
from std.pathlib import Path

from .collection import CollectionManifest
from .collection_store import load_collection_manifest
from .ids import SegmentId
from .paths import collection_segment_root
from .reclaim import (
    CollectionReclaimPlan,
    SNAPSHOT_RETENTION_REASON_ACTIVE,
    SNAPSHOT_RETENTION_REASON_GENERATION,
    SNAPSHOT_RETENTION_REASON_PINNED,
    SNAPSHOT_RETENTION_REASON_RECLAIM,
    SnapshotRetentionDecision,
    SnapshotRetentionPolicy,
)
from .segment_store import load_sealed_segment_manifest
from .snapshot import SnapshotManifest
from .snapshot_store import load_snapshot_manifest, snapshot_manifest_exists


def string_list_contains(read values: List[String], target: String) -> Bool:
    for value in values:
        if value == target:
            return True

    return False


def append_unique_string(mut values: List[String], target: String) -> Bool:
    if string_list_contains(values, target):
        return False

    values.append(target.copy())
    return True


def snapshot_id_is_pinned(
    read policy: SnapshotRetentionPolicy, snapshot_id: String
) -> Bool:
    for pinned_snapshot_id in policy.pinned_snapshot_ids:
        if pinned_snapshot_id.value == snapshot_id:
            return True

    return False


def ordered_snapshots_descending(
    read snapshots: List[SnapshotManifest]
) raises -> List[SnapshotManifest]:
    var ordered = List[SnapshotManifest]()
    var used = List[Bool]()
    for _ in range(len(snapshots)):
        used.append(False)

    for _ in range(len(snapshots)):
        var best_index = -1
        for index in range(len(snapshots)):
            if used[index]:
                continue

            if best_index == -1:
                best_index = index
                continue

            if snapshots[index].generation > snapshots[best_index].generation:
                best_index = index
                continue

            if (
                snapshots[index].generation == snapshots[best_index].generation
                and snapshots[index].snapshot_id.value
                    > snapshots[best_index].snapshot_id.value
            ):
                best_index = index

        if best_index == -1:
            raise Error("failed to order snapshots")

        used[best_index] = True
        ordered.append(snapshots[best_index].copy())

    return ordered^


def require_collection_snapshot_matches_collection(
    read collection: CollectionManifest, read snapshot: SnapshotManifest
) raises:
    if snapshot.collection_id.value != collection.collection_id.value:
        raise Error("snapshot collection_id does not match collection manifest")
    if snapshot.tenant_id.value != collection.tenant_id.value:
        raise Error("snapshot tenant_id does not match collection manifest")
    if snapshot.namespace_id.value != collection.namespace_id.value:
        raise Error("snapshot namespace_id does not match collection manifest")


def load_collection_snapshots(
    collection_root: Path, read collection: CollectionManifest
) raises -> List[SnapshotManifest]:
    var snapshots = List[SnapshotManifest]()
    var snapshots_root = collection_root / "snapshots"
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        return snapshots^

    for snapshot_entry in snapshots_root.listdir():
        var snapshot_root = snapshots_root / snapshot_entry
        if not snapshot_root.is_dir():
            continue
        if not snapshot_manifest_exists(snapshot_root):
            continue

        var snapshot = load_snapshot_manifest(snapshot_root)
        require_collection_snapshot_matches_collection(collection, snapshot)
        snapshots.append(snapshot.copy())

    return ordered_snapshots_descending(snapshots)


def require_active_snapshot_id(
    read collection: CollectionManifest, read snapshots: List[SnapshotManifest]
) raises -> String:
    if collection.active_snapshot_id.byte_length() != 0:
        for snapshot in snapshots:
            if snapshot.snapshot_id.value == collection.active_snapshot_id:
                if snapshot.generation != collection.latest_generation:
                    raise Error(
                        "active_snapshot_id generation does not match latest_generation"
                    )
                return collection.active_snapshot_id.copy()

        raise Error("active_snapshot_id does not resolve to a stored snapshot")

    if collection.latest_generation == 0:
        return ""

    var active_snapshot_id = String()
    var active_snapshot_count = 0
    for snapshot in snapshots:
        if snapshot.generation != collection.latest_generation:
            continue

        active_snapshot_id = snapshot.snapshot_id.value.copy()
        active_snapshot_count += 1

    if active_snapshot_count == 0:
        raise Error("latest_generation does not match any stored snapshot")
    if active_snapshot_count != 1:
        raise Error("latest_generation matches multiple stored snapshots")

    return active_snapshot_id^


def build_collection_reclaim_plan(
    collection_root: Path, read policy: SnapshotRetentionPolicy
) raises -> CollectionReclaimPlan:
    var collection = load_collection_manifest(collection_root)
    var snapshots = load_collection_snapshots(collection_root, collection)
    var active_snapshot_id = require_active_snapshot_id(collection, snapshots)
    var retained_snapshot_ids = List[String]()
    var decisions = List[SnapshotRetentionDecision]()

    if active_snapshot_id.byte_length() != 0:
        retained_snapshot_ids.append(active_snapshot_id.copy())

    var retained_generation_count = 0
    for snapshot in snapshots:
        var snapshot_id = snapshot.snapshot_id.value
        if snapshot_id == active_snapshot_id:
            continue
        if snapshot_id_is_pinned(policy, snapshot_id):
            _ = append_unique_string(retained_snapshot_ids, snapshot_id)
            continue
        if retained_generation_count >= policy.keep_latest_inactive_count:
            continue

        _ = append_unique_string(retained_snapshot_ids, snapshot_id)
        retained_generation_count += 1

    var retained_segment_ids = List[String]()
    var reclaimable_unique_segment_ids = List[SegmentId]()
    var reclaimable_unique_byte_size = 0
    var inactive_snapshot_count = 0
    var retained_inactive_snapshot_count = 0
    var reclaimable_snapshot_count = 0

    for snapshot in snapshots:
        if string_list_contains(retained_snapshot_ids, snapshot.snapshot_id.value):
            for segment_id in snapshot.segment_ids:
                _ = append_unique_string(retained_segment_ids, segment_id.value)

    for snapshot in snapshots:
        var retain = string_list_contains(
            retained_snapshot_ids, snapshot.snapshot_id.value
        )
        var reason = String(SNAPSHOT_RETENTION_REASON_ACTIVE)
        if snapshot.snapshot_id.value == active_snapshot_id:
            pass
        elif snapshot_id_is_pinned(policy, snapshot.snapshot_id.value):
            reason = SNAPSHOT_RETENTION_REASON_PINNED
            inactive_snapshot_count += 1
            retained_inactive_snapshot_count += 1
        elif retain:
            reason = SNAPSHOT_RETENTION_REASON_GENERATION
            inactive_snapshot_count += 1
            retained_inactive_snapshot_count += 1
        else:
            reason = SNAPSHOT_RETENTION_REASON_RECLAIM
            inactive_snapshot_count += 1
            reclaimable_snapshot_count += 1
            for segment_id in snapshot.segment_ids:
                if string_list_contains(retained_segment_ids, segment_id.value):
                    continue

                var appended = False
                if not string_list_contains(
                    retained_segment_ids, segment_id.value
                ):
                    appended = True
                    for existing in reclaimable_unique_segment_ids:
                        if existing.value == segment_id.value:
                            appended = False
                            break

                if appended:
                    reclaimable_unique_segment_ids.append(segment_id.copy())
                    reclaimable_unique_byte_size += load_sealed_segment_manifest(
                        collection_segment_root(collection_root, segment_id)
                    ).stats.byte_size

        decisions.append(
            SnapshotRetentionDecision(
                snapshot.snapshot_id,
                snapshot.generation,
                snapshot.stats.segment_count,
                snapshot.stats.byte_size,
                retain,
                reason,
            )
        )

    return CollectionReclaimPlan(
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        active_snapshot_id,
        len(snapshots),
        inactive_snapshot_count,
        retained_inactive_snapshot_count,
        reclaimable_snapshot_count,
        len(reclaimable_unique_segment_ids),
        reclaimable_unique_byte_size,
        decisions,
        reclaimable_unique_segment_ids,
    )
