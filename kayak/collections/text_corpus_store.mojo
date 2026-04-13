from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.storage.atomic_write import write_text_atomic
from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import append_line, parse_int, read_non_empty_lines, split_tab_fields
from kayak.text import DocumentTextCorpus

from .artifact_manifest import (
    read_collection_artifact_manifest,
    write_collection_artifact_manifest,
)
from .ids import CollectionId, SegmentId
from .paths import (
    text_corpus_entries_path,
    text_corpus_manifest_path,
    text_corpus_payload_root,
)
from .text_corpus import StoredDocumentTextCorpus


def stored_document_text_corpus_exists(root: Path) -> Bool:
    return text_corpus_manifest_path(root).exists()


def save_stored_document_text_corpus(
    root: Path, read stored: StoredDocumentTextCorpus
) raises:
    makedirs(root, exist_ok=True)
    makedirs(text_corpus_payload_root(root), exist_ok=True)

    write_collection_artifact_manifest(
        text_corpus_manifest_path(root),
        "document_text_corpus",
        [
            ManifestEntry("collection_id", stored.collection_id.value),
            ManifestEntry("segment_id", stored.segment_id.value),
            ManifestEntry("payload_encoding", "file_per_doc_utf8"),
            ManifestEntry("document_count", String(len(stored.corpus.doc_ids))),
        ],
    )

    # This baseline favors exact text preservation over packed-string efficiency.
    # A later artifact can replace the payload layout without changing the contract.
    var entry_lines = String()
    var payload_root = text_corpus_payload_root(root)
    for index in range(len(stored.corpus.doc_ids)):
        var file_name = String(index) + ".txt"
        append_line(entry_lines, stored.corpus.doc_ids[index] + "\t" + file_name)
        write_text_atomic(payload_root / file_name, stored.corpus.texts[index])

    write_text_atomic(text_corpus_entries_path(root), entry_lines)


def load_stored_document_text_corpus(
    root: Path
) raises -> StoredDocumentTextCorpus:
    var entries = read_collection_artifact_manifest(
        text_corpus_manifest_path(root), "document_text_corpus"
    )

    if require_manifest_value(entries, "payload_encoding") != "file_per_doc_utf8":
        raise Error("unsupported document text corpus payload encoding")

    var doc_ids = List[String]()
    var texts = List[String]()

    for line in read_non_empty_lines(text_corpus_entries_path(root)):
        var fields = split_tab_fields(line, 2, "document text corpus entry")
        doc_ids.append(fields[0])
        texts.append((text_corpus_payload_root(root) / fields[1]).read_text())

    if len(doc_ids) != parse_int(
        require_manifest_value(entries, "document_count"), "document_count"
    ):
        raise Error(
            "stored document text corpus document_count does not match payload"
        )

    return StoredDocumentTextCorpus(
        CollectionId(require_manifest_value(entries, "collection_id")),
        SegmentId(require_manifest_value(entries, "segment_id")),
        DocumentTextCorpus(doc_ids^, texts^),
    )
