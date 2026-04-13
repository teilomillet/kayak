# Internal filter-scope composition for hosted collection identity.
#
# This module owns how request-visible filter expressions are combined with the
# logical collection scope when a segment carries scope-aware filter postings.
# It does not own planner availability or filter-index storage.

from kayak.collections import (
    CollectionManifest,
    collection_layout_family_is_shared_pool,
    LoadedSealedSegment,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX,
    loaded_segment_has_document_filter_index,
    stored_document_filter_index_has_logical_scope_postings,
)
from kayak.filters import (
    FilterExpression,
    LogicalFilterScope,
    conjoin_filter_expressions,
    logical_scope_filter,
)


def logical_filter_scope_for_collection(
    read collection: CollectionManifest
) raises -> LogicalFilterScope:
    return LogicalFilterScope(
        collection.collection_id.value,
        collection.tenant_id.value,
        collection.namespace_id.value,
    )


def collection_requires_logical_scope_pushdown(
    read collection: CollectionManifest
) -> Bool:
    return collection_layout_family_is_shared_pool(
        collection.collection_layout_family
    )


def segment_supports_scoped_filter_pushdown(
    read segment: LoadedSealedSegment
) -> Bool:
    if not loaded_segment_has_document_filter_index(segment):
        return False

    # Capability checks should stay non-raising so planner/execution code can
    # ask whether scoped pushdown is available without treating missing support
    # as an exceptional path.
    for artifact in segment.search_artifacts:
        if artifact.manifest.family != SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX:
            continue

        return stored_document_filter_index_has_logical_scope_postings(
            artifact.stored_document_filter_index
        )

    return False


def effective_filter_expression_for_collection(
    read collection: CollectionManifest,
    read filter_expression: FilterExpression,
) raises -> FilterExpression:
    if not collection_requires_logical_scope_pushdown(collection):
        return filter_expression.copy()

    var scope_filter = logical_scope_filter(
        logical_filter_scope_for_collection(collection)
    )
    if filter_expression.is_match_all():
        return scope_filter.copy()

    return conjoin_filter_expressions(filter_expression, scope_filter)


def effective_filter_expression_for_segment(
    read collection: CollectionManifest,
    read segment: LoadedSealedSegment,
    read filter_expression: FilterExpression,
) raises -> FilterExpression:
    var effective_filter = effective_filter_expression_for_collection(
        collection,
        filter_expression,
    )
    if not collection_requires_logical_scope_pushdown(collection):
        return effective_filter.copy()

    if not segment_supports_scoped_filter_pushdown(segment):
        raise Error(
            "shared_pool collections require a scope-aware document_filter_index sidecar on every segment"
        )

    return effective_filter.copy()
