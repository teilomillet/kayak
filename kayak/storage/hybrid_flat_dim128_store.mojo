from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.memory import Span

from kayak.index import HybridFlatDim128Index, build_hybrid_flat_dim128_index
from kayak.numeric import STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM

from .binary_int_codec import read_non_negative_int_payload_with_encoding
from .binary_vector_codec import (
    read_binary_scalar_payload,
    write_binary_scalar_payload,
)
from .manifest import (
    ManifestEntry,
    load_optional_manifest_value,
    read_manifest,
    require_manifest_value,
    require_supported_storage_format,
    write_manifest,
)
from .metadata import StoredHybridFlatDim128Index, StoredPackedIndex
from .text_codec import append_line, parse_int, read_non_empty_lines
from .vector_payload_encoding import VECTOR_PAYLOAD_ENCODING_BINARY_LE

# Owns persistence for the optional hybrid flat dim128 artifact.
# It does not choose when callers should use this layout.
struct HybridFlatDim128CacheEntry(Copyable):
    var stored_index: StoredHybridFlatDim128Index
    var loaded_from_storage: Bool

    def __init__(
        out self,
        var stored_index: StoredHybridFlatDim128Index,
        loaded_from_storage: Bool,
    ):
        self.stored_index = stored_index^
        self.loaded_from_storage = loaded_from_storage


def hybrid_flat_dim128_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def hybrid_flat_dim128_index_exists(root: Path) -> Bool:
    return hybrid_flat_dim128_manifest_path(root).exists()


def hybrid_flat_dim128_default_root(root: Path) -> Path:
    return root / "hybrid_flat_dim128_index"


def build_stored_hybrid_flat_dim128_index(
    read stored_packed_index: StoredPackedIndex
) raises -> StoredHybridFlatDim128Index:
    return StoredHybridFlatDim128Index(
        stored_packed_index.dataset_id.copy(),
        stored_packed_index.model_name.copy(),
        stored_packed_index.vector_scalar_name.copy(),
        build_hybrid_flat_dim128_index(stored_packed_index.index),
    )


def packed_storage_supports_direct_hybrid_flat_dim128_materialization(
    packed_root: Path
) raises -> Bool:
    var manifest = read_manifest(packed_root / "manifest.tsv")
    var format_version = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != "packed_index":
        return False

    if parse_int(require_manifest_value(manifest, "vector_dim"), "vector_dim") != COLBERT_VECTOR_DIM:
        return False

    if format_version < 2:
        return False

    return (
        require_manifest_value(manifest, "vector_payload_encoding")
        == VECTOR_PAYLOAD_ENCODING_BINARY_LE
        and (packed_root / "token_vectors.bin").exists()
    )


def write_hybrid_flat_dim128_doc_offsets_from_packed_storage(
    hybrid_root: Path,
    packed_root: Path,
    read packed_manifest: List[ManifestEntry],
) raises:
    var doc_offsets_path = hybrid_root / "doc_offsets.tsv"
    if (packed_root / "doc_offsets.tsv").exists():
        doc_offsets_path.write_text((packed_root / "doc_offsets.tsv").read_text())
        return

    var doc_offsets_encoding = load_optional_manifest_value(
        packed_manifest, "doc_offsets_encoding"
    )
    if doc_offsets_encoding.byte_length() == 0:
        raise Error(
            "packed storage is missing both doc_offsets.tsv and doc_offsets_encoding"
        )

    var doc_offsets = read_non_negative_int_payload_with_encoding(
        packed_root / "doc_offsets.bin",
        doc_offsets_encoding,
    )
    var doc_offset_lines = String()
    for doc_offset in doc_offsets:
        append_line(doc_offset_lines, String(doc_offset))
    doc_offsets_path.write_text(doc_offset_lines)


def materialize_stored_hybrid_flat_dim128_index_from_packed_storage(
    hybrid_root: Path, packed_root: Path
) raises:
    var packed_manifest = read_manifest(packed_root / "manifest.tsv")
    _ = require_supported_storage_format(packed_manifest)

    if require_manifest_value(packed_manifest, "artifact_kind") != "packed_index":
        raise Error("storage artifact is not a packed index")

    if (
        parse_int(require_manifest_value(packed_manifest, "vector_dim"), "vector_dim")
        != COLBERT_VECTOR_DIM
    ):
        raise Error("packed storage does not use vector_dim=128")

    if not packed_storage_supports_direct_hybrid_flat_dim128_materialization(
        packed_root
    ):
        raise Error(
            "packed storage does not support direct hybrid flat dim128 materialization"
        )

    makedirs(hybrid_root, exist_ok=True)

    write_manifest(
        hybrid_flat_dim128_manifest_path(hybrid_root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "hybrid_flat_dim128_index"),
            ManifestEntry(
                "vector_scalar_name",
                require_manifest_value(packed_manifest, "vector_scalar_name"),
            ),
            ManifestEntry(
                "dataset_id",
                require_manifest_value(packed_manifest, "dataset_id"),
            ),
            ManifestEntry(
                "model_name",
                require_manifest_value(packed_manifest, "model_name"),
            ),
            ManifestEntry("vector_payload_encoding", "binary_le"),
            ManifestEntry("vector_dim", String(COLBERT_VECTOR_DIM)),
            ManifestEntry(
                "document_count",
                require_manifest_value(packed_manifest, "document_count"),
            ),
            ManifestEntry(
                "total_vector_count",
                require_manifest_value(packed_manifest, "total_vector_count"),
            ),
        ],
    )

    (hybrid_root / "doc_ids.tsv").write_text((packed_root / "doc_ids.tsv").read_text())
    write_hybrid_flat_dim128_doc_offsets_from_packed_storage(
        hybrid_root,
        packed_root,
        packed_manifest,
    )
    var token_value_bytes = (packed_root / "token_vectors.bin").read_bytes()
    (hybrid_root / "token_values.bin").write_bytes(Span(token_value_bytes))


def save_stored_hybrid_flat_dim128_index(
    root: Path, stored: StoredHybridFlatDim128Index
) raises:
    makedirs(root, exist_ok=True)

    write_manifest(
        hybrid_flat_dim128_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "hybrid_flat_dim128_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", "binary_le"),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("document_count", String(stored.index.document_count)),
            ManifestEntry(
                "total_vector_count", String(stored.index.total_vector_count)
            ),
        ],
    )

    var doc_id_lines = String()
    for doc_id in stored.index.doc_ids:
        append_line(doc_id_lines, doc_id)
    (root / "doc_ids.tsv").write_text(doc_id_lines)

    var doc_offset_lines = String()
    for doc_offset in stored.index.doc_offsets:
        append_line(doc_offset_lines, String(doc_offset))
    (root / "doc_offsets.tsv").write_text(doc_offset_lines)

    write_binary_scalar_payload(root / "token_values.bin", stored.index.token_values)


def load_stored_hybrid_flat_dim128_index(
    root: Path
) raises -> StoredHybridFlatDim128Index:
    var manifest = read_manifest(hybrid_flat_dim128_manifest_path(root))
    _ = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != "hybrid_flat_dim128_index":
        raise Error("storage artifact is not a hybrid flat dim128 index")

    if require_manifest_value(manifest, "vector_payload_encoding") != "binary_le":
        raise Error("unsupported hybrid flat dim128 payload encoding")

    var vector_dim = parse_int(
        require_manifest_value(manifest, "vector_dim"), "vector_dim"
    )
    if vector_dim != COLBERT_VECTOR_DIM:
        raise Error("stored hybrid flat dim128 index requires vector_dim=128")

    var doc_ids = read_non_empty_lines(root / "doc_ids.tsv")
    var doc_offsets = List[Int]()
    for line in read_non_empty_lines(root / "doc_offsets.tsv"):
        doc_offsets.append(parse_int(line, "doc offset"))

    var token_values = read_binary_scalar_payload(root / "token_values.bin")
    var index = HybridFlatDim128Index(
        doc_ids^,
        doc_offsets^,
        token_values^,
        vector_dim,
    )

    if index.document_count != parse_int(
        require_manifest_value(manifest, "document_count"), "document_count"
    ):
        raise Error("stored hybrid flat dim128 document_count does not match payload")

    if index.total_vector_count != parse_int(
        require_manifest_value(manifest, "total_vector_count"),
        "total_vector_count",
    ):
        raise Error(
            "stored hybrid flat dim128 total_vector_count does not match payload"
        )

    return StoredHybridFlatDim128Index(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        index^,
    )


def ensure_stored_hybrid_flat_dim128_index(
    root: Path, read stored_packed_index: StoredPackedIndex
) raises -> HybridFlatDim128CacheEntry:
    if hybrid_flat_dim128_index_exists(root):
        return HybridFlatDim128CacheEntry(
            load_stored_hybrid_flat_dim128_index(root), True
        )

    var stored_hybrid_index = build_stored_hybrid_flat_dim128_index(
        stored_packed_index
    )
    save_stored_hybrid_flat_dim128_index(root, stored_hybrid_index.copy())
    return HybridFlatDim128CacheEntry(stored_hybrid_index^, False)
