from std.os import makedirs
from std.pathlib import Path

from kayak.numeric import STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME

from .manifest import (
    ManifestEntry,
    read_manifest,
    require_manifest_value,
    require_supported_storage_format,
    write_manifest,
)
from .metadata import StoredGemGraphIndex
from .text_codec import parse_int


def gem_graph_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def gem_graph_index_exists(root: Path) -> Bool:
    return gem_graph_manifest_path(root).exists()


def gem_graph_storage_byte_size(root: Path) raises -> Int:
    return gem_graph_manifest_path(root).read_text().byte_length()


def save_stored_gem_graph_index(root: Path, read stored: StoredGemGraphIndex) raises:
    makedirs(root, exist_ok=True)
    write_manifest(
        gem_graph_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "gem_graph_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("document_count", String(stored.document_count)),
            ManifestEntry("cluster_count", String(stored.cluster_count)),
            ManifestEntry("graph_edge_count", String(stored.graph_edge_count)),
            ManifestEntry(
                "shortcut_edge_count", String(stored.shortcut_edge_count)
            ),
            ManifestEntry("entry_point_count", String(stored.entry_point_count)),
            ManifestEntry(
                "quantization_centroid_count",
                String(stored.quantization_centroid_count),
            ),
            ManifestEntry("artifact_byte_size", "0"),
        ],
    )

    var artifact_byte_size = gem_graph_storage_byte_size(root)
    write_manifest(
        gem_graph_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "gem_graph_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("document_count", String(stored.document_count)),
            ManifestEntry("cluster_count", String(stored.cluster_count)),
            ManifestEntry("graph_edge_count", String(stored.graph_edge_count)),
            ManifestEntry(
                "shortcut_edge_count", String(stored.shortcut_edge_count)
            ),
            ManifestEntry("entry_point_count", String(stored.entry_point_count)),
            ManifestEntry(
                "quantization_centroid_count",
                String(stored.quantization_centroid_count),
            ),
            ManifestEntry("artifact_byte_size", String(artifact_byte_size)),
        ],
    )


def load_stored_gem_graph_index(root: Path) raises -> StoredGemGraphIndex:
    var manifest = read_manifest(gem_graph_manifest_path(root))
    _ = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != "gem_graph_index":
        raise Error("storage artifact is not a gem graph index")

    return StoredGemGraphIndex(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        parse_int(require_manifest_value(manifest, "document_count"), "document_count"),
        parse_int(require_manifest_value(manifest, "cluster_count"), "cluster_count"),
        parse_int(
            require_manifest_value(manifest, "graph_edge_count"),
            "graph_edge_count",
        ),
        parse_int(
            require_manifest_value(manifest, "shortcut_edge_count"),
            "shortcut_edge_count",
        ),
        parse_int(
            require_manifest_value(manifest, "entry_point_count"),
            "entry_point_count",
        ),
        parse_int(
            require_manifest_value(manifest, "quantization_centroid_count"),
            "quantization_centroid_count",
        ),
        parse_int(
            require_manifest_value(manifest, "artifact_byte_size"),
            "artifact_byte_size",
        ),
    )
