from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import parse_int

from .artifact_manifest import (
    read_collection_artifact_manifest,
    require_current_vector_scalar_name,
    write_collection_artifact_manifest,
)
from .collection import CollectionManifest
from .ids import CollectionId, NamespaceId, TenantId
from .manifest_util import load_optional_manifest_value
from .paths import collection_manifest_path


def collection_manifest_exists(root: Path) -> Bool:
    return collection_manifest_path(root).exists()


def save_collection_manifest(root: Path, read manifest: CollectionManifest) raises:
    makedirs(root, exist_ok=True)

    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("collection_id", manifest.collection_id.value))
    entries.append(ManifestEntry("tenant_id", manifest.tenant_id.value))
    entries.append(ManifestEntry("namespace_id", manifest.namespace_id.value))
    entries.append(ManifestEntry("model_name", manifest.model_name))
    entries.append(
        ManifestEntry("vector_scalar_name", manifest.vector_scalar_name)
    )
    entries.append(ManifestEntry("vector_dim", String(manifest.vector_dim)))
    entries.append(
        ManifestEntry("latest_generation", String(manifest.latest_generation))
    )
    if manifest.active_snapshot_id.byte_length() != 0:
        entries.append(
            ManifestEntry("active_snapshot_id", manifest.active_snapshot_id)
        )

    write_collection_artifact_manifest(
        collection_manifest_path(root), "collection_manifest", entries
    )


def load_collection_manifest(root: Path) raises -> CollectionManifest:
    var entries = read_collection_artifact_manifest(
        collection_manifest_path(root), "collection_manifest"
    )

    return CollectionManifest(
        CollectionId(require_manifest_value(entries, "collection_id")),
        TenantId(require_manifest_value(entries, "tenant_id")),
        NamespaceId(require_manifest_value(entries, "namespace_id")),
        require_manifest_value(entries, "model_name"),
        require_current_vector_scalar_name(entries, "collection manifest"),
        parse_int(require_manifest_value(entries, "vector_dim"), "vector_dim"),
        parse_int(
            require_manifest_value(entries, "latest_generation"),
            "latest_generation",
        ),
        load_optional_manifest_value(entries, "active_snapshot_id"),
    )
