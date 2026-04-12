from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.index import CentroidPostingIndex, build_centroid_posting_index
from kayak.numeric import STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME

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
from .metadata import StoredCentroidPostingIndex, StoredPackedIndex
from .text_codec import append_line, parse_int, read_non_empty_lines
from .vector_payload_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    require_supported_packed_index_vector_payload_encoding,
)


comptime CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC = "weight_desc_doc_asc"
comptime CENTROID_POSTINGS_ORDER_UNSPECIFIED = "unspecified"


struct CentroidPostingCacheEntry(Copyable):
    var stored_index: StoredCentroidPostingIndex
    var loaded_from_storage: Bool

    def __init__(
        out self,
        var stored_index: StoredCentroidPostingIndex,
        loaded_from_storage: Bool,
    ):
        self.stored_index = stored_index^
        self.loaded_from_storage = loaded_from_storage


def centroid_postings_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def centroid_document_counts_path(root: Path) -> Path:
    return root / "centroid_document_counts.tsv"


def centroid_token_counts_path(root: Path) -> Path:
    return root / "centroid_token_counts.tsv"


def centroid_postings_index_exists(root: Path) -> Bool:
    return centroid_postings_manifest_path(root).exists()


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def centroid_postings_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "centroid_dims.tsv")
    total += file_size_bytes(root / "posting_offsets.tsv")
    total += file_size_bytes(root / "posting_doc_indices.tsv")
    total += file_size_bytes(root / "posting_weights.tsv")
    total += file_size_bytes(root / "centroid_vectors.bin")
    if centroid_document_counts_path(root).exists():
        total += file_size_bytes(centroid_document_counts_path(root))
    if centroid_token_counts_path(root).exists():
        total += file_size_bytes(centroid_token_counts_path(root))
    return total


def build_stored_centroid_posting_index(
    read stored_packed_index: StoredPackedIndex, centroid_budget: Int
) raises -> StoredCentroidPostingIndex:
    return StoredCentroidPostingIndex(
        stored_packed_index.dataset_id.copy(),
        stored_packed_index.model_name.copy(),
        stored_packed_index.vector_scalar_name.copy(),
        CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC,
        centroid_budget,
        0,
        0,
        build_centroid_posting_index(stored_packed_index.index, centroid_budget),
    )


def write_int_lines(path: Path, read values: List[Int]) raises:
    var lines = String()
    for value in values:
        append_line(lines, String(value))
    path.write_text(lines)


def read_int_lines(path: Path, name: String) raises -> List[Int]:
    var values = List[Int]()
    for line in read_non_empty_lines(path):
        values.append(parse_int(line, name))
    return values^


def load_optional_manifest_value(read entries: List[ManifestEntry], key: String) -> String:
    for entry in entries:
        if entry.key == key:
            return entry.value.copy()

    return ""


def write_centroid_postings_manifest(
    root: Path,
    read stored: StoredCentroidPostingIndex,
    artifact_kind: String,
    vector_payload_encoding: String,
    artifact_byte_size: Int,
) raises:
    write_manifest(
        centroid_postings_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", artifact_kind),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", vector_payload_encoding),
            ManifestEntry("posting_order_kind", stored.posting_order_kind),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("document_count", String(stored.index.document_count)),
            ManifestEntry("centroid_budget", String(stored.centroid_budget)),
            ManifestEntry("posting_cap", String(stored.posting_cap)),
            ManifestEntry("centroid_count", String(stored.index.centroid_count)),
            ManifestEntry(
                "total_posting_count", String(stored.index.total_posting_count)
            ),
            ManifestEntry("artifact_byte_size", String(artifact_byte_size)),
        ],
    )


def write_centroid_postings_manifest(
    root: Path,
    read stored: StoredCentroidPostingIndex,
    vector_payload_encoding: String,
    artifact_byte_size: Int,
) raises:
    write_centroid_postings_manifest(
        root,
        stored,
        "centroid_posting_index",
        vector_payload_encoding,
        artifact_byte_size,
    )


def save_stored_centroid_posting_index_with_artifact_kind(
    root: Path,
    read stored: StoredCentroidPostingIndex,
    artifact_kind: String,
    vector_payload_encoding: String = VECTOR_PAYLOAD_ENCODING_BINARY_LE,
) raises:
    require_supported_packed_index_vector_payload_encoding(vector_payload_encoding)
    makedirs(root, exist_ok=True)

    write_centroid_postings_manifest(
        root, stored, artifact_kind, vector_payload_encoding, 0
    )

    write_int_lines(root / "centroid_dims.tsv", stored.index.centroid_dims)
    write_int_lines(
        centroid_document_counts_path(root),
        stored.index.centroid_document_counts,
    )
    write_int_lines(
        centroid_token_counts_path(root),
        stored.index.centroid_token_counts,
    )
    write_int_lines(root / "posting_offsets.tsv", stored.index.posting_offsets)
    write_int_lines(root / "posting_doc_indices.tsv", stored.index.posting_doc_indices)
    write_int_lines(root / "posting_weights.tsv", stored.index.posting_weights)
    write_binary_vector_payload_with_encoding(
        root / "centroid_vectors.bin",
        stored.index.centroid_vectors,
        vector_payload_encoding,
    )

    var artifact_byte_size = centroid_postings_storage_byte_size(root)
    while True:
        write_centroid_postings_manifest(
            root,
            stored,
            artifact_kind,
            vector_payload_encoding,
            artifact_byte_size,
        )
        var stabilized = centroid_postings_storage_byte_size(root)
        if stabilized == artifact_byte_size:
            break
        artifact_byte_size = stabilized


def save_stored_centroid_posting_index(
    root: Path,
    read stored: StoredCentroidPostingIndex,
    vector_payload_encoding: String = VECTOR_PAYLOAD_ENCODING_BINARY_LE,
) raises:
    save_stored_centroid_posting_index_with_artifact_kind(
        root,
        stored,
        "centroid_posting_index",
        vector_payload_encoding,
    )


def load_stored_centroid_posting_index_with_artifact_kind(
    root: Path, expected_artifact_kind: String
) raises -> StoredCentroidPostingIndex:
    var manifest = read_manifest(centroid_postings_manifest_path(root))
    _ = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != expected_artifact_kind:
        raise Error("storage artifact is not a " + expected_artifact_kind)

    var vector_dim = parse_int(
        require_manifest_value(manifest, "vector_dim"), "vector_dim"
    )
    var document_count = parse_int(
        require_manifest_value(manifest, "document_count"), "document_count"
    )
    var vector_payload_encoding = require_manifest_value(
        manifest, "vector_payload_encoding"
    )
    var posting_order_kind = load_optional_manifest_value(
        manifest, "posting_order_kind"
    )
    if posting_order_kind.byte_length() == 0:
        posting_order_kind = CENTROID_POSTINGS_ORDER_UNSPECIFIED
    var posting_cap_text = load_optional_manifest_value(manifest, "posting_cap")
    var posting_cap = 0
    if posting_cap_text.byte_length() != 0:
        posting_cap = parse_int(posting_cap_text, "posting_cap")
    var centroid_dims = read_int_lines(root / "centroid_dims.tsv", "centroid_dim")
    var posting_offsets = read_int_lines(root / "posting_offsets.tsv", "posting_offset")
    var posting_doc_indices = read_int_lines(
        root / "posting_doc_indices.tsv", "posting_doc_index"
    )
    var posting_weights = read_int_lines(root / "posting_weights.tsv", "posting_weight")
    var centroid_vectors = read_binary_vector_payload_with_encoding(
        root / "centroid_vectors.bin", vector_dim, vector_payload_encoding
    )
    var has_centroid_document_counts = centroid_document_counts_path(root).exists()
    var has_centroid_token_counts = centroid_token_counts_path(root).exists()

    if has_centroid_document_counts != has_centroid_token_counts:
        raise Error(
            "centroid posting index summary files must either both exist or both be absent"
        )

    if has_centroid_document_counts:
        return StoredCentroidPostingIndex(
            require_manifest_value(manifest, "dataset_id"),
            require_manifest_value(manifest, "model_name"),
            VECTOR_SCALAR_NAME,
            posting_order_kind.copy(),
            parse_int(
                require_manifest_value(manifest, "centroid_budget"),
                "centroid_budget",
            ),
            posting_cap,
            parse_int(
                require_manifest_value(manifest, "artifact_byte_size"),
                "artifact_byte_size",
            ),
            CentroidPostingIndex(
                centroid_dims^,
                centroid_vectors^,
                posting_offsets^,
                posting_doc_indices^,
                posting_weights^,
                read_int_lines(
                    centroid_document_counts_path(root), "centroid_document_count"
                ),
                read_int_lines(
                    centroid_token_counts_path(root), "centroid_token_count"
                ),
                vector_dim,
                document_count,
            ),
        )

    return StoredCentroidPostingIndex(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        posting_order_kind,
        parse_int(require_manifest_value(manifest, "centroid_budget"), "centroid_budget"),
        posting_cap,
        parse_int(
            require_manifest_value(manifest, "artifact_byte_size"),
            "artifact_byte_size",
        ),
        CentroidPostingIndex(
            centroid_dims^,
            centroid_vectors^,
            posting_offsets^,
            posting_doc_indices^,
            posting_weights^,
            vector_dim,
            document_count,
        ),
    )


def load_stored_centroid_posting_index(
    root: Path
) raises -> StoredCentroidPostingIndex:
    return load_stored_centroid_posting_index_with_artifact_kind(
        root, "centroid_posting_index"
    )


def ensure_stored_centroid_posting_index(
    root: Path,
    read stored_packed_index: StoredPackedIndex,
    centroid_budget: Int,
) raises -> CentroidPostingCacheEntry:
    if centroid_postings_index_exists(root):
        var loaded = load_stored_centroid_posting_index(root)
        if (
            loaded.centroid_budget == centroid_budget
            and loaded.dataset_id == stored_packed_index.dataset_id
            and loaded.model_name == stored_packed_index.model_name
            and loaded.posting_order_kind
                == CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC
            and loaded.index.document_count
                == stored_packed_index.index.document_count
            and loaded.index.vector_dim == stored_packed_index.index.vector_dim
            and centroid_document_counts_path(root).exists()
            and centroid_token_counts_path(root).exists()
        ):
            return CentroidPostingCacheEntry(loaded.copy(), True)

    var stored_centroid_index = build_stored_centroid_posting_index(
        stored_packed_index, centroid_budget
    )
    save_stored_centroid_posting_index(root, stored_centroid_index.copy())
    return CentroidPostingCacheEntry(
        load_stored_centroid_posting_index(root), False
    )
