from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionManifest,
    collection_manifest_exists,
    load_collection_manifest,
    load_sealed_segment_manifest,
    load_snapshot_manifest,
    snapshot_manifest_exists,
)
from kayak.collections.paths import collection_segment_root

from .draft_state import load_draft_state_metadata
from .paths import draft_state_root, service_collections_root
from .service_status import ServiceHealthStatus, ServiceMetricsSnapshot


def service_collection_roots(service_root: Path) raises -> List[Path]:
    var collection_roots = List[Path]()
    var collections_root = service_collections_root(service_root)
    if not collections_root.exists() or not collections_root.is_dir():
        return collection_roots^

    for tenant_entry in collections_root.listdir():
        var tenant_root = collections_root / tenant_entry
        if not tenant_root.is_dir():
            continue

        for namespace_entry in tenant_root.listdir():
            var namespace_root = tenant_root / namespace_entry
            if not namespace_root.is_dir():
                continue

            for collection_entry in namespace_root.listdir():
                var collection_root = namespace_root / collection_entry
                if not collection_root.is_dir():
                    continue

                if collection_manifest_exists(collection_root):
                    collection_roots.append(collection_root)

    return collection_roots^


def require_snapshot_matches_collection(
    read collection: CollectionManifest, snapshot_root: Path
) raises:
    var snapshot = load_snapshot_manifest(snapshot_root)
    if snapshot.collection_id.value != collection.collection_id.value:
        raise Error("live snapshot collection_id does not match collection manifest")

    if snapshot.tenant_id.value != collection.tenant_id.value:
        raise Error("live snapshot tenant_id does not match collection manifest")

    if snapshot.namespace_id.value != collection.namespace_id.value:
        raise Error("live snapshot namespace_id does not match collection manifest")


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


def snapshot_roots_for_collection(
    collection_root: Path, read collection: CollectionManifest
) raises -> List[Path]:
    var snapshot_roots = List[Path]()
    var snapshots_root = collection_root / "snapshots"
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        return snapshot_roots^

    for snapshot_entry in snapshots_root.listdir():
        var snapshot_root = snapshots_root / snapshot_entry
        if not snapshot_root.is_dir():
            continue
        if not snapshot_manifest_exists(snapshot_root):
            continue

        require_snapshot_matches_collection(collection, snapshot_root)
        snapshot_roots.append(snapshot_root)

    return snapshot_roots^


def live_snapshot_root_for_collection(
    collection_root: Path, read collection: CollectionManifest
) raises -> Path:
    if collection.active_snapshot_id.byte_length() != 0:
        var active_root = collection_root / "snapshots" / collection.active_snapshot_id
        if not snapshot_manifest_exists(active_root):
            raise Error("collection active_snapshot_id does not resolve to a snapshot")

        var active_snapshot = load_snapshot_manifest(active_root)
        require_snapshot_matches_collection(collection, active_root)
        if active_snapshot.generation != collection.latest_generation:
            raise Error(
                "collection active_snapshot_id generation does not match latest_generation"
            )

        return active_root

    if collection.latest_generation == 0:
        raise Error("collection does not have a live snapshot generation")

    var snapshots_root = collection_root / "snapshots"
    if not snapshots_root.exists() or not snapshots_root.is_dir():
        raise Error("collection latest_generation has no snapshots directory")

    var live_snapshot_root = Path("")
    var live_snapshot_count = 0

    for snapshot_entry in snapshots_root.listdir():
        var snapshot_root = snapshots_root / snapshot_entry
        if not snapshot_root.is_dir():
            continue
        if not snapshot_manifest_exists(snapshot_root):
            continue

        var snapshot = load_snapshot_manifest(snapshot_root)
        if snapshot.generation != collection.latest_generation:
            continue

        require_snapshot_matches_collection(collection, snapshot_root)
        live_snapshot_root = snapshot_root
        live_snapshot_count += 1
    if live_snapshot_count == 0:
        raise Error(
            "collection latest_generation does not match any snapshot manifest"
        )
    if live_snapshot_count != 1:
        raise Error("collection latest_generation matches multiple snapshot manifests")

    return live_snapshot_root


def build_service_metrics_snapshot(service_root: Path) raises -> ServiceMetricsSnapshot:
    var collection_count = 0
    var segment_count = 0
    var document_count = 0
    var vector_count = 0
    var byte_size = 0
    var published_snapshot_count = 0
    var inactive_snapshot_count = 0
    var inactive_unique_segment_count = 0
    var inactive_unique_byte_size = 0
    var pending_draft_collection_count = 0
    var pending_draft_mutation_count = 0

    for collection_root in service_collection_roots(service_root):
        collection_count += 1
        var collection = load_collection_manifest(collection_root)
        var draft_metadata = load_draft_state_metadata(
            draft_state_root(collection_root), collection
        )
        if draft_metadata.mutation_count > 0:
            pending_draft_collection_count += 1
            pending_draft_mutation_count += draft_metadata.mutation_count

        var active_snapshot_root = Path("")
        var active_segment_ids = List[String]()
        if collection.latest_generation == 0:
            _ = active_segment_ids
        else:
            active_snapshot_root = live_snapshot_root_for_collection(
                collection_root, collection
            )
            var live_snapshot = load_snapshot_manifest(active_snapshot_root)
            segment_count += live_snapshot.stats.segment_count
            document_count += live_snapshot.stats.document_count
            vector_count += live_snapshot.stats.total_vector_count
            byte_size += live_snapshot.stats.byte_size
            for segment_id in live_snapshot.segment_ids:
                active_segment_ids.append(segment_id.value.copy())

        var inactive_unique_segment_ids = List[String]()
        for snapshot_root in snapshot_roots_for_collection(collection_root, collection):
            published_snapshot_count += 1
            var is_active_snapshot = False
            if collection.latest_generation != 0:
                is_active_snapshot = (
                    snapshot_root.__fspath__() == active_snapshot_root.__fspath__()
                )

            if is_active_snapshot:
                continue

            inactive_snapshot_count += 1
            var snapshot = load_snapshot_manifest(snapshot_root)
            for segment_id in snapshot.segment_ids:
                if string_list_contains(active_segment_ids, segment_id.value):
                    continue

                if append_unique_string(
                    inactive_unique_segment_ids, segment_id.value
                ):
                    var segment = load_sealed_segment_manifest(
                        collection_segment_root(collection_root, segment_id)
                    )
                    inactive_unique_segment_count += 1
                    inactive_unique_byte_size += segment.stats.byte_size

    return ServiceMetricsSnapshot(
        collection_count,
        segment_count,
        document_count,
        vector_count,
        byte_size,
        published_snapshot_count,
        inactive_snapshot_count,
        inactive_unique_segment_count,
        inactive_unique_byte_size,
        pending_draft_collection_count,
        pending_draft_mutation_count,
    )


def build_service_health_status(service_root: Path) raises -> ServiceHealthStatus:
    var collection_count = 0
    var live_snapshot_count = 0

    for collection_root in service_collection_roots(service_root):
        collection_count += 1
        var collection = load_collection_manifest(collection_root)
        if collection.latest_generation == 0:
            continue

        _ = live_snapshot_root_for_collection(collection_root, collection)
        live_snapshot_count += 1

    return ServiceHealthStatus(
        "ok",
        collection_count,
        live_snapshot_count,
        0,
    )
