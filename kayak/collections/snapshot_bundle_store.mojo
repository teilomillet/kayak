from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import parse_int

from .artifact_manifest import (
    read_collection_artifact_manifest,
    write_collection_artifact_manifest,
)
from .ids import CollectionId, NamespaceId, SnapshotId, TenantId
from .snapshot_bundle import SnapshotExportBundleManifest


def snapshot_bundle_manifest_path(root: Path) -> Path:
    return root / "snapshot_bundle.manifest.tsv"


def snapshot_export_bundle_manifest_exists(root: Path) -> Bool:
    return snapshot_bundle_manifest_path(root).exists()


def save_snapshot_export_bundle_manifest(
    root: Path, read manifest: SnapshotExportBundleManifest
) raises:
    makedirs(root, exist_ok=True)

    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("snapshot_id", manifest.snapshot_id.value))
    entries.append(ManifestEntry("collection_id", manifest.collection_id.value))
    entries.append(ManifestEntry("tenant_id", manifest.tenant_id.value))
    entries.append(ManifestEntry("namespace_id", manifest.namespace_id.value))
    entries.append(ManifestEntry("generation", String(manifest.generation)))
    entries.append(ManifestEntry("segment_count", String(manifest.segment_count)))

    write_collection_artifact_manifest(
        snapshot_bundle_manifest_path(root), "snapshot_export_bundle", entries
    )


def load_snapshot_export_bundle_manifest(
    root: Path
) raises -> SnapshotExportBundleManifest:
    var entries = read_collection_artifact_manifest(
        snapshot_bundle_manifest_path(root), "snapshot_export_bundle"
    )

    return SnapshotExportBundleManifest(
        SnapshotId(require_manifest_value(entries, "snapshot_id")),
        CollectionId(require_manifest_value(entries, "collection_id")),
        TenantId(require_manifest_value(entries, "tenant_id")),
        NamespaceId(require_manifest_value(entries, "namespace_id")),
        parse_int(require_manifest_value(entries, "generation"), "generation"),
        parse_int(
            require_manifest_value(entries, "segment_count"), "segment_count"
        ),
    )
