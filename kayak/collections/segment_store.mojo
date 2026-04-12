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
from .ids import CollectionId, NamespaceId, SegmentId, TenantId
from .paths import require_relative_artifact_root, segment_manifest_path
from .segment import SealedSegmentManifest
from .stats_manifest import (
    load_segment_stats_from_manifest,
    segment_stats_manifest_entries,
)


def sealed_segment_manifest_exists(root: Path) -> Bool:
    return segment_manifest_path(root).exists()


def encode_optional_text_corpus_root(text_corpus_root: String) -> String:
    if text_corpus_root.byte_length() == 0:
        return "-"

    return text_corpus_root.copy()


def decode_optional_text_corpus_root(text_corpus_root: String) -> String:
    if text_corpus_root == "-":
        return ""

    return text_corpus_root.copy()


def save_sealed_segment_manifest(
    root: Path, read manifest: SealedSegmentManifest
) raises:
    makedirs(root, exist_ok=True)
    var packed_index_root = require_relative_artifact_root(
        manifest.packed_index_root, "packed_index_root"
    )
    var text_corpus_root = String()
    if manifest.text_corpus_root.byte_length() != 0:
        text_corpus_root = require_relative_artifact_root(
            manifest.text_corpus_root, "text_corpus_root"
        )

    var entries = List[ManifestEntry]()
    entries.append(ManifestEntry("segment_id", manifest.segment_id.value))
    entries.append(ManifestEntry("collection_id", manifest.collection_id.value))
    entries.append(ManifestEntry("tenant_id", manifest.tenant_id.value))
    entries.append(ManifestEntry("namespace_id", manifest.namespace_id.value))
    entries.append(ManifestEntry("generation", String(manifest.generation)))
    entries.append(ManifestEntry("model_name", manifest.model_name))
    entries.append(
        ManifestEntry("vector_scalar_name", manifest.vector_scalar_name)
    )
    entries.append(ManifestEntry("vector_dim", String(manifest.vector_dim)))
    entries.append(ManifestEntry("packed_index_root", packed_index_root))
    entries.append(
        ManifestEntry(
            "text_corpus_root",
            encode_optional_text_corpus_root(text_corpus_root),
        )
    )

    for entry in segment_stats_manifest_entries(manifest.stats):
        entries.append(entry.copy())

    write_collection_artifact_manifest(
        segment_manifest_path(root), "sealed_segment_manifest", entries
    )


def load_sealed_segment_manifest(root: Path) raises -> SealedSegmentManifest:
    var entries = read_collection_artifact_manifest(
        segment_manifest_path(root), "sealed_segment_manifest"
    )

    return SealedSegmentManifest(
        SegmentId(require_manifest_value(entries, "segment_id")),
        CollectionId(require_manifest_value(entries, "collection_id")),
        TenantId(require_manifest_value(entries, "tenant_id")),
        NamespaceId(require_manifest_value(entries, "namespace_id")),
        parse_int(require_manifest_value(entries, "generation"), "generation"),
        require_manifest_value(entries, "model_name"),
        require_current_vector_scalar_name(entries, "sealed segment manifest"),
        parse_int(require_manifest_value(entries, "vector_dim"), "vector_dim"),
        require_manifest_value(entries, "packed_index_root"),
        decode_optional_text_corpus_root(
            require_manifest_value(entries, "text_corpus_root")
        ),
        load_segment_stats_from_manifest(entries),
    )
