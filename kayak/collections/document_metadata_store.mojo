from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.storage.atomic_write import write_text_atomic
from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import append_line, parse_int, read_non_empty_lines, split_tab_fields

from .artifact_manifest import (
    read_collection_artifact_manifest,
    write_collection_artifact_manifest,
)
from .document_metadata import (
    DocumentMetadataEntry,
    DocumentMetadataMap,
    StoredDocumentMetadataCorpus,
)
from .ids import CollectionId, SegmentId
from .paths import (
    document_metadata_entries_path,
    document_metadata_manifest_path,
    document_metadata_payload_root,
)


def stored_document_metadata_corpus_exists(root: Path) -> Bool:
    return document_metadata_manifest_path(root).exists()


def save_stored_document_metadata_corpus(
    root: Path, read stored: StoredDocumentMetadataCorpus
) raises:
    makedirs(root, exist_ok=True)
    makedirs(document_metadata_payload_root(root), exist_ok=True)

    write_collection_artifact_manifest(
        document_metadata_manifest_path(root),
        "document_metadata_corpus",
        [
            ManifestEntry("collection_id", stored.collection_id.value),
            ManifestEntry("segment_id", stored.segment_id.value),
            ManifestEntry("payload_encoding", "file_per_doc_tsv"),
            ManifestEntry("document_count", String(stored.document_count())),
        ],
    )

    var entry_lines = String()
    var payload_root = document_metadata_payload_root(root)
    for index in range(stored.document_count()):
        var file_name = String(index) + ".tsv"
        append_line(entry_lines, stored.doc_ids[index] + "\t" + file_name)

        var payload = String()
        for entry in stored.metadata_maps[index].entries:
            append_line(payload, entry.key + "\t" + entry.value)

        write_text_atomic(payload_root / file_name, payload)

    write_text_atomic(document_metadata_entries_path(root), entry_lines)


def load_stored_document_metadata_corpus(
    root: Path
) raises -> StoredDocumentMetadataCorpus:
    var entries = read_collection_artifact_manifest(
        document_metadata_manifest_path(root), "document_metadata_corpus"
    )

    if require_manifest_value(entries, "payload_encoding") != "file_per_doc_tsv":
        raise Error("unsupported document metadata corpus payload encoding")

    var doc_ids = List[String]()
    var metadata_maps = List[DocumentMetadataMap]()

    for line in read_non_empty_lines(document_metadata_entries_path(root)):
        var fields = split_tab_fields(line, 2, "document metadata corpus entry")
        doc_ids.append(fields[0])

        var metadata_entries = List[DocumentMetadataEntry]()
        for metadata_line in read_non_empty_lines(
            document_metadata_payload_root(root) / fields[1]
        ):
            var metadata_fields = split_tab_fields(
                metadata_line, 2, "document metadata entry"
            )
            metadata_entries.append(
                DocumentMetadataEntry(
                    metadata_fields[0],
                    metadata_fields[1],
                )
            )

        metadata_maps.append(DocumentMetadataMap(metadata_entries))
    if len(doc_ids) != parse_int(
        require_manifest_value(entries, "document_count"),
        "document metadata document_count",
    ):
        raise Error(
            "stored document metadata corpus document_count does not match payload"
        )

    return StoredDocumentMetadataCorpus(
        CollectionId(require_manifest_value(entries, "collection_id")),
        SegmentId(require_manifest_value(entries, "segment_id")),
        doc_ids,
        metadata_maps,
    )
