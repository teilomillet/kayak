from std.pathlib import Path

from .collection import CollectionManifest
from .collection_store import save_collection_manifest
from .snapshot import SnapshotManifest
from .snapshot_store import save_snapshot_manifest


def publish_snapshot_manifest(
    collection_root: Path, read snapshot: SnapshotManifest
) raises:
    save_snapshot_manifest(collection_root / "snapshots" / snapshot.snapshot_id.value, snapshot)


def publish_collection_snapshot(
    collection_root: Path,
    read collection: CollectionManifest,
    read snapshot: SnapshotManifest,
) raises -> CollectionManifest:
    if snapshot.generation < collection.latest_generation:
        raise Error("published snapshot generation must be monotonic")

    publish_snapshot_manifest(collection_root, snapshot)
    var published = CollectionManifest(
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        collection.model_name.copy(),
        collection.vector_scalar_name.copy(),
        collection.vector_dim,
        snapshot.generation,
        snapshot.snapshot_id.value.copy(),
    )
    save_collection_manifest(collection_root, published)
    return published^


def promote_collection_generation(
    collection_root: Path, read collection: CollectionManifest, generation: Int
) raises -> CollectionManifest:
    var published = CollectionManifest(
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        collection.model_name.copy(),
        collection.vector_scalar_name.copy(),
        collection.vector_dim,
        generation,
        collection.active_snapshot_id.copy(),
    )
    save_collection_manifest(collection_root, published)
    return published^
