from std.collections import List
from std.os import remove, rmdir
from std.pathlib import Path

from .collection import CollectionManifest
from .collection_store import load_collection_manifest
from .ids import SegmentId, SnapshotId
from .paths import collection_segment_root, collection_snapshot_root
from .reclaim import (
    CollectionReclaimExecutionResult,
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


def remove_tree(path: Path) raises:
    if not path.exists():
        return

    if path.is_dir():
        for child in path.listdir():
            remove_tree(path / child)
        rmdir(path)
        return

    remove(path)


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


def find_snapshot_decision_index(
    read decisions: List[SnapshotRetentionDecision], snapshot_id: String
) -> Int:
    for index in range(len(decisions)):
        if decisions[index].snapshot_id.value == snapshot_id:
            return index

    return -1


def append_unique_segment_id(mut segment_ids: List[SegmentId], read segment_id: SegmentId) -> Bool:
    for existing in segment_ids:
        if existing.value == segment_id.value:
            return False

    segment_ids.append(segment_id.copy())
    return True


def retain_snapshot_ids_for_plan(read plan: CollectionReclaimPlan) -> List[String]:
    var retained_snapshot_ids = List[String]()
    for decision in plan.decisions:
        if decision.retain:
            retained_snapshot_ids.append(decision.snapshot_id.value.copy())

    return retained_snapshot_ids^


def reclaim_snapshot_ids_for_plan(read plan: CollectionReclaimPlan) -> List[SnapshotId]:
    var reclaimable_snapshot_ids = List[SnapshotId]()
    for decision in plan.decisions:
        if not decision.retain:
            reclaimable_snapshot_ids.append(decision.snapshot_id.copy())

    return reclaimable_snapshot_ids^


def require_collection_matches_reclaim_plan(
    read collection: CollectionManifest, read plan: CollectionReclaimPlan
) raises:
    if collection.collection_id.value != plan.collection_id.value:
        raise Error("reclaim plan collection_id does not match collection manifest")
    if collection.tenant_id.value != plan.tenant_id.value:
        raise Error("reclaim plan tenant_id does not match collection manifest")
    if collection.namespace_id.value != plan.namespace_id.value:
        raise Error("reclaim plan namespace_id does not match collection manifest")


def require_reclaim_plan_matches_collection_state(
    collection_root: Path,
    read collection: CollectionManifest,
    read plan: CollectionReclaimPlan,
) raises:
    require_collection_matches_reclaim_plan(collection, plan)

    var snapshots = load_collection_snapshots(collection_root, collection)
    var active_snapshot_id = require_active_snapshot_id(collection, snapshots)
    if active_snapshot_id != plan.active_snapshot_id:
        raise Error("reclaim plan active_snapshot_id does not match current collection state")
    if len(snapshots) != plan.total_snapshot_count:
        raise Error("reclaim plan total_snapshot_count does not match current collection state")

    var retained_snapshot_ids = retain_snapshot_ids_for_plan(plan)
    var retained_segment_ids = List[String]()
    var current_reclaimable_segment_ids = List[SegmentId]()
    var current_reclaimable_byte_size = 0
    var inactive_snapshot_count = 0
    var retained_inactive_snapshot_count = 0
    var reclaimable_snapshot_count = 0

    for snapshot in snapshots:
        var decision_index = find_snapshot_decision_index(
            plan.decisions, snapshot.snapshot_id.value
        )
        if decision_index == -1:
            raise Error("reclaim plan is missing a current snapshot decision")

        var decision = plan.decisions[decision_index].copy()
        if decision.generation != snapshot.generation:
            raise Error("reclaim plan snapshot generation does not match current state")
        if decision.segment_count != snapshot.stats.segment_count:
            raise Error("reclaim plan snapshot segment_count does not match current state")
        if decision.byte_size != snapshot.stats.byte_size:
            raise Error("reclaim plan snapshot byte_size does not match current state")

        if decision.snapshot_id.value != active_snapshot_id:
            inactive_snapshot_count += 1
            if decision.retain:
                retained_inactive_snapshot_count += 1
            else:
                reclaimable_snapshot_count += 1
        elif not decision.retain:
            raise Error("reclaim plan cannot mark the active snapshot reclaimable")

        if string_list_contains(retained_snapshot_ids, snapshot.snapshot_id.value):
            for segment_id in snapshot.segment_ids:
                _ = append_unique_string(retained_segment_ids, segment_id.value)

    for snapshot in snapshots:
        var decision_index = find_snapshot_decision_index(
            plan.decisions, snapshot.snapshot_id.value
        )
        var decision = plan.decisions[decision_index].copy()
        if decision.retain:
            continue

        for segment_id in snapshot.segment_ids:
            if string_list_contains(retained_segment_ids, segment_id.value):
                continue
            if append_unique_segment_id(current_reclaimable_segment_ids, segment_id):
                current_reclaimable_byte_size += load_sealed_segment_manifest(
                    collection_segment_root(collection_root, segment_id)
                ).stats.byte_size

    if inactive_snapshot_count != plan.inactive_snapshot_count:
        raise Error("reclaim plan inactive_snapshot_count does not match current state")
    if retained_inactive_snapshot_count != plan.retained_inactive_snapshot_count:
        raise Error(
            "reclaim plan retained_inactive_snapshot_count does not match current state"
        )
    if reclaimable_snapshot_count != plan.reclaimable_snapshot_count:
        raise Error("reclaim plan reclaimable_snapshot_count does not match current state")
    if len(current_reclaimable_segment_ids) != plan.reclaimable_unique_segment_count:
        raise Error(
            "reclaim plan reclaimable_unique_segment_count does not match current state"
        )
    if current_reclaimable_byte_size != plan.reclaimable_unique_byte_size:
        raise Error(
            "reclaim plan reclaimable_unique_byte_size does not match current state"
        )
    if len(current_reclaimable_segment_ids) != len(plan.reclaimable_unique_segment_ids):
        raise Error(
            "reclaim plan reclaimable_unique_segment_ids length does not match current state"
        )

    for segment_id in current_reclaimable_segment_ids:
        var found = False
        for planned_segment_id in plan.reclaimable_unique_segment_ids:
            if planned_segment_id.value == segment_id.value:
                found = True
                break
        if not found:
            raise Error(
                "reclaim plan reclaimable_unique_segment_ids do not match current state"
            )


def execute_collection_reclaim_plan(
    collection_root: Path,
    read plan: CollectionReclaimPlan,
    dry_run: Bool = True,
) raises -> CollectionReclaimExecutionResult:
    var collection = load_collection_manifest(collection_root)
    require_reclaim_plan_matches_collection_state(collection_root, collection, plan)

    var reclaimable_snapshot_ids = reclaim_snapshot_ids_for_plan(plan)
    var reclaimable_segment_ids = plan.reclaimable_unique_segment_ids.copy()

    if not dry_run:
        for snapshot_id in reclaimable_snapshot_ids:
            remove_tree(collection_snapshot_root(collection_root, snapshot_id))

        var remaining_snapshots = load_collection_snapshots(collection_root, collection)
        var retained_segment_ids = List[String]()
        for snapshot in remaining_snapshots:
            for segment_id in snapshot.segment_ids:
                _ = append_unique_string(retained_segment_ids, segment_id.value)

        for segment_id in reclaimable_segment_ids:
            if string_list_contains(retained_segment_ids, segment_id.value):
                raise Error(
                    "refusing to delete a segment still referenced by a retained snapshot"
                )

            remove_tree(collection_segment_root(collection_root, segment_id))

    return CollectionReclaimExecutionResult(
        plan.collection_id,
        plan.tenant_id,
        plan.namespace_id,
        not dry_run,
        len(reclaimable_snapshot_ids),
        len(reclaimable_segment_ids),
        plan.reclaimable_unique_byte_size,
        reclaimable_snapshot_ids,
        reclaimable_segment_ids,
    )


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
