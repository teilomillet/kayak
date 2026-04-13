from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionManifest,
    collection_manifest_exists,
    load_collection_manifest,
    load_snapshot_manifest,
    snapshot_manifest_exists,
)

from .paths import service_collections_root
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


def live_snapshot_root_for_collection(
    collection_root: Path, read collection: CollectionManifest
) raises -> Path:
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

    for collection_root in service_collection_roots(service_root):
        collection_count += 1
        var collection = load_collection_manifest(collection_root)
        if collection.latest_generation == 0:
            continue

        var live_snapshot = load_snapshot_manifest(
            live_snapshot_root_for_collection(collection_root, collection)
        )
        segment_count += live_snapshot.stats.segment_count
        document_count += live_snapshot.stats.document_count
        vector_count += live_snapshot.stats.total_vector_count
        byte_size += live_snapshot.stats.byte_size

    return ServiceMetricsSnapshot(
        collection_count,
        segment_count,
        document_count,
        vector_count,
        byte_size,
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
