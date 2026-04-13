from kayak.collections import (
    LoadedSealedSegment,
    StoredDocumentFilterIndex,
    loaded_segment_has_document_filter_index,
    loaded_segment_stored_document_filter_index,
)
from kayak.collections.document_filter_runtime import (
    DocumentFilterAllowlist,
    document_filter_allowlist_for_expression,
)
from kayak.filters import (
    FilterExpression,
    filter_expression_requires_document_metadata,
)


def empty_document_filter_index_for_segment(
    read segment: LoadedSealedSegment
) raises -> StoredDocumentFilterIndex:
    return StoredDocumentFilterIndex(
        segment.manifest.collection_id,
        segment.manifest.segment_id,
        segment.stored_index.index.document_count,
        0,
        [],
    )


def document_filter_allowlist_for_segment(
    read segment: LoadedSealedSegment,
    read filter_expression: FilterExpression,
) raises -> DocumentFilterAllowlist:
    if filter_expression_requires_document_metadata(filter_expression):
        if not loaded_segment_has_document_filter_index(segment):
            raise Error(
                "structured filters require a document filter index sidecar on every segment"
            )

        return document_filter_allowlist_for_expression(
            segment.stored_index.index.doc_ids,
            loaded_segment_stored_document_filter_index(segment),
            filter_expression,
        )

    return document_filter_allowlist_for_expression(
        segment.stored_index.index.doc_ids,
        empty_document_filter_index_for_segment(segment),
        filter_expression,
    )


def document_filter_allowlist_artifact_byte_size_for_segment(
    read segment: LoadedSealedSegment,
    read filter_expression: FilterExpression,
) raises -> Int:
    if not filter_expression_requires_document_metadata(filter_expression):
        return 0

    if not loaded_segment_has_document_filter_index(segment):
        raise Error(
            "structured filters require a document filter index sidecar on every segment"
        )

    return loaded_segment_stored_document_filter_index(
        segment
    ).artifact_byte_size
