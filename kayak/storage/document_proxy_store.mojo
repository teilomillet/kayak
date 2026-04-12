from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.index import DocumentProxyIndex, build_document_proxy_index
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
from .metadata import StoredDocumentProxyIndex, StoredPackedIndex
from .text_codec import append_line, parse_int, read_non_empty_lines
from .vector_payload_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    require_supported_packed_index_vector_payload_encoding,
)


struct DocumentProxyCacheEntry(Copyable):
    var stored_index: StoredDocumentProxyIndex
    var loaded_from_storage: Bool

    def __init__(
        out self,
        var stored_index: StoredDocumentProxyIndex,
        loaded_from_storage: Bool,
    ):
        self.stored_index = stored_index^
        self.loaded_from_storage = loaded_from_storage


def document_proxy_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def document_proxy_index_exists(root: Path) -> Bool:
    return document_proxy_manifest_path(root).exists()


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def document_proxy_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    total += file_size_bytes(root / "proxy_vectors.bin")
    return total


def build_stored_document_proxy_index(
    read stored_packed_index: StoredPackedIndex, document_vector_budget: Int
) raises -> StoredDocumentProxyIndex:
    return StoredDocumentProxyIndex(
        stored_packed_index.dataset_id.copy(),
        stored_packed_index.model_name.copy(),
        stored_packed_index.vector_scalar_name.copy(),
        document_vector_budget,
        1,
        0,
        build_document_proxy_index(
            stored_packed_index.index, document_vector_budget
        ),
    )


def save_stored_document_proxy_index(
    root: Path,
    read stored: StoredDocumentProxyIndex,
    vector_payload_encoding: String = VECTOR_PAYLOAD_ENCODING_BINARY_LE,
) raises:
    require_supported_packed_index_vector_payload_encoding(vector_payload_encoding)
    makedirs(root, exist_ok=True)

    write_manifest(
        document_proxy_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "document_proxy_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", vector_payload_encoding),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("document_count", String(stored.index.document_count)),
            ManifestEntry(
                "document_vector_budget", String(stored.document_vector_budget)
            ),
            ManifestEntry(
                "proxy_vector_count_per_document",
                String(stored.proxy_vector_count_per_document),
            ),
            ManifestEntry(
                "total_vector_count", String(stored.index.document_count)
            ),
            ManifestEntry("artifact_byte_size", "0"),
        ],
    )

    var doc_id_lines = String()
    for doc_id in stored.index.doc_ids:
        append_line(doc_id_lines, doc_id)
    (root / "doc_ids.tsv").write_text(doc_id_lines)

    write_binary_vector_payload_with_encoding(
        root / "proxy_vectors.bin",
        stored.index.proxy_vectors,
        vector_payload_encoding,
    )

    var artifact_byte_size = document_proxy_storage_byte_size(root)
    write_manifest(
        document_proxy_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "document_proxy_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", vector_payload_encoding),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("document_count", String(stored.index.document_count)),
            ManifestEntry(
                "document_vector_budget", String(stored.document_vector_budget)
            ),
            ManifestEntry(
                "proxy_vector_count_per_document",
                String(stored.proxy_vector_count_per_document),
            ),
            ManifestEntry(
                "total_vector_count", String(stored.index.document_count)
            ),
            ManifestEntry("artifact_byte_size", String(artifact_byte_size)),
        ],
    )


def load_stored_document_proxy_index(
    root: Path
) raises -> StoredDocumentProxyIndex:
    var manifest = read_manifest(document_proxy_manifest_path(root))
    _ = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != "document_proxy_index":
        raise Error("storage artifact is not a document proxy index")

    var vector_dim = parse_int(
        require_manifest_value(manifest, "vector_dim"), "vector_dim"
    )
    var vector_payload_encoding = require_manifest_value(
        manifest, "vector_payload_encoding"
    )
    var doc_ids = read_non_empty_lines(root / "doc_ids.tsv")
    var proxy_vectors = read_binary_vector_payload_with_encoding(
        root / "proxy_vectors.bin", vector_dim, vector_payload_encoding
    )

    if len(doc_ids) != len(proxy_vectors):
        raise Error("stored document proxy doc_ids length does not match payload")

    return StoredDocumentProxyIndex(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        parse_int(
            require_manifest_value(manifest, "document_vector_budget"),
            "document_vector_budget",
        ),
        parse_int(
            require_manifest_value(
                manifest, "proxy_vector_count_per_document"
            ),
            "proxy_vector_count_per_document",
        ),
        parse_int(
            require_manifest_value(manifest, "artifact_byte_size"),
            "artifact_byte_size",
        ),
        DocumentProxyIndex(doc_ids^, proxy_vectors^, vector_dim),
    )


def ensure_stored_document_proxy_index(
    root: Path,
    read stored_packed_index: StoredPackedIndex,
    document_vector_budget: Int,
) raises -> DocumentProxyCacheEntry:
    if document_proxy_index_exists(root):
        var loaded = load_stored_document_proxy_index(root)
        if loaded.document_vector_budget == document_vector_budget:
            return DocumentProxyCacheEntry(loaded.copy(), True)

    var stored_document_proxy = build_stored_document_proxy_index(
        stored_packed_index, document_vector_budget
    )
    save_stored_document_proxy_index(root, stored_document_proxy.copy())
    return DocumentProxyCacheEntry(
        load_stored_document_proxy_index(root), False
    )
