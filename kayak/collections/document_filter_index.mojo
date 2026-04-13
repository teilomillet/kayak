# Segment-local exact allowlists for structured metadata filters.
#
# This module owns the persisted logical representation of field/value -> doc-index
# postings. It does not own query-time filter evaluation or search-plan policy.

from std.collections import List

from .document_metadata import DocumentMetadataMap
from .ids import CollectionId, SegmentId
from .validation import require_non_empty_string, require_non_negative_int


struct DocumentFilterPosting(Copyable):
    var field_name: String
    var value: String
    var doc_indices: List[Int]

    def __init__(
        out self,
        var field_name: String,
        var value: String,
        read doc_indices: List[Int],
    ) raises:
        self.field_name = require_non_empty_string(
            field_name, "document filter posting field_name"
        )
        self.value = require_non_empty_string(
            value, "document filter posting value"
        )
        var validated_doc_indices = List[Int]()
        var previous_doc_index = -1
        for doc_index in doc_indices:
            var normalized = require_non_negative_int(
                doc_index, "document filter posting doc_index"
            )
            if previous_doc_index != -1 and normalized <= previous_doc_index:
                raise Error(
                    "document filter posting doc_indices must be strictly ascending"
                )
            validated_doc_indices.append(normalized)
            previous_doc_index = normalized

        self.doc_indices = validated_doc_indices^


struct StoredDocumentFilterIndex(Copyable):
    var collection_id: CollectionId
    var segment_id: SegmentId
    var document_count: Int
    var artifact_byte_size: Int
    var postings: List[DocumentFilterPosting]

    def __init__(
        out self,
        collection_id: CollectionId,
        segment_id: SegmentId,
        document_count: Int,
        artifact_byte_size: Int,
        read postings: List[DocumentFilterPosting],
    ) raises:
        self.collection_id = collection_id.copy()
        self.segment_id = segment_id.copy()
        self.document_count = require_non_negative_int(
            document_count, "stored document filter index document_count"
        )
        self.artifact_byte_size = require_non_negative_int(
            artifact_byte_size, "stored document filter index artifact_byte_size"
        )
        var validated_postings = List[DocumentFilterPosting]()
        for posting in postings:
            for doc_index in posting.doc_indices:
                if doc_index >= self.document_count:
                    raise Error(
                        "document filter posting doc_index exceeds document_count"
                    )
            validated_postings.append(posting.copy())

        self.postings = validated_postings^

    def posting_count(self) -> Int:
        return len(self.postings)

    def doc_indices_for(self, field_name: String, value: String) -> List[Int]:
        for posting in self.postings:
            if posting.field_name == field_name and posting.value == value:
                return posting.doc_indices.copy()

        return List[Int]()


def find_document_filter_posting_index(
    read postings: List[DocumentFilterPosting], field_name: String, value: String
) -> Int:
    for index in range(len(postings)):
        if postings[index].field_name == field_name and postings[index].value == value:
            return index

    return -1


def append_doc_index_to_filter_posting(
    mut postings: List[DocumentFilterPosting],
    field_name: String,
    value: String,
    doc_index: Int,
) raises:
    var posting_index = find_document_filter_posting_index(postings, field_name, value)
    if posting_index == -1:
        postings.append(
            DocumentFilterPosting(
                field_name,
                value,
                [doc_index],
            )
        )
        return

    var doc_indices = postings[posting_index].doc_indices.copy()
    if len(doc_indices) != 0 and doc_indices[len(doc_indices) - 1] == doc_index:
        return

    doc_indices.append(doc_index)
    postings[posting_index] = DocumentFilterPosting(
        postings[posting_index].field_name.copy(),
        postings[posting_index].value.copy(),
        doc_indices,
    )


def build_stored_document_filter_index(
    collection_id: CollectionId,
    segment_id: SegmentId,
    read metadata_maps: List[DocumentMetadataMap],
    artifact_byte_size: Int = 0,
) raises -> StoredDocumentFilterIndex:
    var postings = List[DocumentFilterPosting]()

    for doc_index in range(len(metadata_maps)):
        for entry in metadata_maps[doc_index].entries:
            append_doc_index_to_filter_posting(
                postings,
                entry.key,
                entry.value,
                doc_index,
            )

    return StoredDocumentFilterIndex(
        collection_id,
        segment_id,
        len(metadata_maps),
        artifact_byte_size,
        postings,
    )
