from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import append_line, parse_int, read_non_empty_lines

from .artifact_manifest import (
    read_collection_artifact_manifest,
    write_collection_artifact_manifest,
)
from .ids import CollectionId, NamespaceId, SegmentId, SnapshotId, TenantId
from .paths import snapshot_manifest_path, snapshot_segment_ids_path
from .snapshot import SnapshotManifest
from .stats_manifest import (
    collection_stats_manifest_entries,
    load_collection_stats_from_manifest,
)


def snapshot_manifest_exists(root: Path) -> Bool:
    return snapshot_manifest_path(root).exists()


def save_snapshot_manifest(root: Path, read manifest: SnapshotManifest) raises:
    makedirs(root, exist_ok=True)

    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("snapshot_id", manifest.snapshot_id.value))
    entries.append(ManifestEntry("collection_id", manifest.collection_id.value))
    entries.append(ManifestEntry("tenant_id", manifest.tenant_id.value))
    entries.append(ManifestEntry("namespace_id", manifest.namespace_id.value))
    entries.append(ManifestEntry("generation", String(manifest.generation)))

    for entry in collection_stats_manifest_entries(manifest.stats):
        entries.append(entry.copy())

    write_collection_artifact_manifest(
        snapshot_manifest_path(root), "snapshot_manifest", entries
    )

    var segment_id_lines = String()
    for segment_id in manifest.segment_ids:
        append_line(segment_id_lines, segment_id.value)
    snapshot_segment_ids_path(root).write_text(segment_id_lines)


def load_snapshot_manifest(root: Path) raises -> SnapshotManifest:
    var entries = read_collection_artifact_manifest(
        snapshot_manifest_path(root), "snapshot_manifest"
    )

    var segment_ids = List[SegmentId]()
    for line in read_non_empty_lines(snapshot_segment_ids_path(root)):
        segment_ids.append(SegmentId(line))

    var stats = load_collection_stats_from_manifest(entries)
    if len(segment_ids) != stats.segment_count:
        raise Error("snapshot segment_ids length does not match segment_count")

    return SnapshotManifest(
        SnapshotId(require_manifest_value(entries, "snapshot_id")),
        CollectionId(require_manifest_value(entries, "collection_id")),
        TenantId(require_manifest_value(entries, "tenant_id")),
        NamespaceId(require_manifest_value(entries, "namespace_id")),
        parse_int(require_manifest_value(entries, "generation"), "generation"),
        segment_ids^,
        stats^,
    )
