from std.collections import List

from .ids import CollectionId, SegmentId
from .validation import require_non_empty_string


def require_inline_document_metadata_text(
    text: String,
    field_name: String,
    allow_empty: Bool = False,
) raises -> String:
    var normalized = text.copy()
    if not allow_empty:
        normalized = require_non_empty_string(text, field_name)

    if normalized.find("\n") != -1 or normalized.find("\r") != -1:
        raise Error(field_name + " must not contain newlines")
    if normalized.find("\t") != -1:
        raise Error(field_name + " must not contain tabs")

    return normalized^


struct DocumentMetadataEntry(Copyable):
    var key: String
    var value: String

    def __init__(out self, key: String, value: String) raises:
        self.key = require_inline_document_metadata_text(
            key, "document metadata key"
        )
        self.value = require_inline_document_metadata_text(
            value, "document metadata value"
        )


struct DocumentMetadataUpdate(Copyable):
    var key: String
    var value: String

    def __init__(out self, key: String, value: String) raises:
        self.key = require_inline_document_metadata_text(
            key, "document metadata update key"
        )
        self.value = require_inline_document_metadata_text(
            value, "document metadata update value", True
        )


struct DocumentMetadataMap(Copyable):
    var entries: List[DocumentMetadataEntry]

    def __init__(out self):
        self.entries = List[DocumentMetadataEntry]()

    def __init__(out self, read entries: List[DocumentMetadataEntry]) raises:
        var validated_entries = List[DocumentMetadataEntry]()
        for entry in entries:
            for existing in validated_entries:
                if existing.key == entry.key:
                    raise Error(
                        "duplicate document metadata key: " + entry.key
                    )

            validated_entries.append(entry.copy())

        self.entries = validated_entries^

    def entry_count(self) -> Int:
        return len(self.entries)

    def is_empty(self) -> Bool:
        return len(self.entries) == 0


struct StoredDocumentMetadataCorpus(Copyable):
    var collection_id: CollectionId
    var segment_id: SegmentId
    var doc_ids: List[String]
    var metadata_maps: List[DocumentMetadataMap]

    def __init__(
        out self,
        collection_id: CollectionId,
        segment_id: SegmentId,
        read doc_ids: List[String],
        read metadata_maps: List[DocumentMetadataMap],
    ) raises:
        if len(doc_ids) != len(metadata_maps):
            raise Error(
                "stored document metadata corpus requires aligned doc_ids and metadata"
            )

        var validated_doc_ids = List[String]()
        for doc_id in doc_ids:
            validated_doc_ids.append(
                require_non_empty_string(doc_id, "document metadata doc_id")
            )

        self.collection_id = collection_id.copy()
        self.segment_id = segment_id.copy()
        self.doc_ids = validated_doc_ids^
        self.metadata_maps = metadata_maps.copy()

    def document_count(self) -> Int:
        return len(self.doc_ids)


def empty_document_metadata_map() -> DocumentMetadataMap:
    return DocumentMetadataMap()


def document_metadata_value(
    read metadata: DocumentMetadataMap, key: String
) -> String:
    for entry in metadata.entries:
        if entry.key == key:
            return entry.value.copy()

    return ""


def document_metadata_has_key(
    read metadata: DocumentMetadataMap, key: String
) -> Bool:
    return document_metadata_value(metadata, key).byte_length() != 0


def copy_document_metadata_maps(
    read metadata_maps: List[DocumentMetadataMap]
) -> List[DocumentMetadataMap]:
    return metadata_maps.copy()


def merge_document_metadata(
    read base: DocumentMetadataMap,
    read updates: List[DocumentMetadataUpdate],
) raises -> DocumentMetadataMap:
    var merged_entries = List[DocumentMetadataEntry]()
    for entry in base.entries:
        merged_entries.append(entry.copy())

    for update in updates:
        var existing_index = -1
        for index in range(len(merged_entries)):
            if merged_entries[index].key == update.key:
                existing_index = index
                break

        if update.value.byte_length() == 0:
            if existing_index == -1:
                continue

            var kept_entries = List[DocumentMetadataEntry]()
            for index in range(len(merged_entries)):
                if index == existing_index:
                    continue
                kept_entries.append(merged_entries[index].copy())

            merged_entries = kept_entries^
            continue

        var next_entry = DocumentMetadataEntry(
            update.key.copy(),
            update.value.copy(),
        )
        if existing_index == -1:
            merged_entries.append(next_entry.copy())
        else:
            merged_entries[existing_index] = next_entry.copy()

    return DocumentMetadataMap(merged_entries)
