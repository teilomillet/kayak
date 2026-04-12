from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.index import PackedIndex
from kayak.numeric import STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME, VectorScalar

from .binary_vector_codec import (
    read_binary_vector_payload_with_encoding,
    write_binary_vector_payload_with_encoding,
)
from .manifest import (
    ManifestEntry,
    read_manifest,
    require_manifest_value,
    require_supported_storage_format,
    write_manifest,
)
from .metadata import StoredPackedIndex
from .text_codec import (
    append_line,
    decode_vector_line,
    parse_int,
    read_non_empty_lines,
)
from .vector_payload_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    require_supported_packed_index_vector_payload_encoding,
)


def packed_index_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def packed_index_exists(root: Path) -> Bool:
    return packed_index_manifest_path(root).exists()


def save_stored_packed_index(root: Path, stored: StoredPackedIndex) raises:
    save_stored_packed_index_with_encoding(
        root, stored, VECTOR_PAYLOAD_ENCODING_BINARY_LE
    )


def save_stored_packed_index_with_encoding(
    root: Path,
    stored: StoredPackedIndex,
    vector_payload_encoding: String,
) raises:
    require_supported_packed_index_vector_payload_encoding(vector_payload_encoding)
    makedirs(root, exist_ok=True)

    write_manifest(
        packed_index_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "packed_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", vector_payload_encoding),
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
    var doc_ids_path = root / "doc_ids.tsv"
    doc_ids_path.write_text(doc_id_lines)

    var doc_offset_lines = String()
    for doc_offset in stored.index.doc_offsets:
        append_line(doc_offset_lines, String(doc_offset))
    var doc_offsets_path = root / "doc_offsets.tsv"
    doc_offsets_path.write_text(doc_offset_lines)

    write_binary_vector_payload_with_encoding(
        root / "token_vectors.bin",
        stored.index.token_vectors,
        vector_payload_encoding,
    )


def load_stored_packed_index(root: Path) raises -> StoredPackedIndex:
    var manifest = read_manifest(packed_index_manifest_path(root))
    var format_version = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != "packed_index":
        raise Error("storage artifact is not a packed index")

    var vector_dim = parse_int(
        require_manifest_value(manifest, "vector_dim"), "vector_dim"
    )
    var doc_ids = read_non_empty_lines(root / "doc_ids.tsv")
    var doc_offsets = List[Int]()
    for line in read_non_empty_lines(root / "doc_offsets.tsv"):
        doc_offsets.append(parse_int(line, "doc offset"))

    var token_vectors = List[List[VectorScalar]]()
    if format_version >= 2:
        var vector_payload_encoding = require_manifest_value(
            manifest, "vector_payload_encoding"
        )
        token_vectors = read_binary_vector_payload_with_encoding(
            root / "token_vectors.bin", vector_dim, vector_payload_encoding
        )
    else:
        for line in read_non_empty_lines(root / "token_vectors.tsv"):
            token_vectors.append(decode_vector_line(line))

    var index = PackedIndex(
        doc_ids^,
        doc_offsets^,
        token_vectors^,
        vector_dim,
    )

    if index.document_count != parse_int(
        require_manifest_value(manifest, "document_count"), "document_count"
    ):
        raise Error("stored packed index document_count does not match payload")

    if index.total_vector_count != parse_int(
        require_manifest_value(manifest, "total_vector_count"),
        "total_vector_count",
    ):
        raise Error("stored packed index total_vector_count does not match payload")

    return StoredPackedIndex(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        index^,
    )
