# Storage codec for the document-filter allowlist sidecar.

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.storage.atomic_write import write_text_atomic
from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import (
    append_line,
    parse_int,
    read_non_empty_lines,
    split_fields,
    split_tab_fields,
)

from .artifact_manifest import (
    read_collection_artifact_manifest,
    write_collection_artifact_manifest,
)
from .document_filter_index import (
    DocumentFilterPosting,
    StoredDocumentFilterIndex,
)
from .ids import CollectionId, SegmentId
from .paths import (
    document_filter_index_entries_path,
    document_filter_index_manifest_path,
)
from .search_artifact import SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX


def stored_document_filter_index_exists(root: Path) -> Bool:
    return document_filter_index_manifest_path(root).exists()


def document_filter_index_storage_byte_size(root: Path) raises -> Int:
    var total = document_filter_index_manifest_path(root).read_text().byte_length()
    total += document_filter_index_entries_path(root).read_text().byte_length()
    return total


def encode_doc_indices_line(read doc_indices: List[Int]) -> String:
    var line = String()
    for index in range(len(doc_indices)):
        if index > 0:
            line += ","
        line += String(doc_indices[index])

    return line^


def decode_doc_indices_line(line: String) raises -> List[Int]:
    if line.byte_length() == 0:
        return []

    var doc_indices = List[Int]()
    for field in split_fields(line, ","):
        doc_indices.append(parse_int(field, "document filter posting doc_index"))

    return doc_indices^


def save_stored_document_filter_index(
    root: Path, read stored: StoredDocumentFilterIndex
) raises:
    makedirs(root, exist_ok=True)

    var entry_lines = String()
    for posting in stored.postings:
        append_line(
            entry_lines,
            posting.field_name + "\t" + posting.value + "\t"
            + encode_doc_indices_line(posting.doc_indices),
        )

    write_text_atomic(document_filter_index_entries_path(root), entry_lines)
    write_collection_artifact_manifest(
        document_filter_index_manifest_path(root),
        SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX,
        [
            ManifestEntry("collection_id", stored.collection_id.value),
            ManifestEntry("segment_id", stored.segment_id.value),
            ManifestEntry("payload_encoding", "tsv_inline_postings"),
            ManifestEntry("document_count", String(stored.document_count)),
            ManifestEntry("posting_count", String(stored.posting_count())),
        ],
    )


def load_stored_document_filter_index(
    root: Path
) raises -> StoredDocumentFilterIndex:
    var entries = read_collection_artifact_manifest(
        document_filter_index_manifest_path(root),
        SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX,
    )

    if require_manifest_value(entries, "payload_encoding") != "tsv_inline_postings":
        raise Error("unsupported document filter index payload encoding")

    var postings = List[DocumentFilterPosting]()
    for line in read_non_empty_lines(document_filter_index_entries_path(root)):
        var fields = split_tab_fields(line, 3, "document filter posting entry")
        postings.append(
            DocumentFilterPosting(
                fields[0],
                fields[1],
                decode_doc_indices_line(fields[2]),
            )
        )

    if len(postings) != parse_int(
        require_manifest_value(entries, "posting_count"),
        "document filter posting_count",
    ):
        raise Error(
            "stored document filter index posting_count does not match payload"
        )

    return StoredDocumentFilterIndex(
        CollectionId(require_manifest_value(entries, "collection_id")),
        SegmentId(require_manifest_value(entries, "segment_id")),
        parse_int(
            require_manifest_value(entries, "document_count"),
            "document filter document_count",
        ),
        document_filter_index_storage_byte_size(root),
        postings,
    )
