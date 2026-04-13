from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    DocumentFilterPosting,
    DocumentMetadataEntry,
    DocumentMetadataMap,
    FILTER_FIELD_INTERNAL_COLLECTION_ID,
    FILTER_FIELD_INTERNAL_NAMESPACE_ID,
    FILTER_FIELD_INTERNAL_TENANT_ID,
    LogicalFilterScope,
    NamespaceId,
    SegmentId,
    StoredDocumentFilterIndex,
    TenantId,
    and_filter,
    build_stored_document_filter_index,
    conjoin_filter_expressions,
    document_filter_allowlist_for_expression,
    filter_expression_matches_document_in_scope,
    logical_scope_filter,
    load_stored_document_filter_index,
    save_stored_document_filter_index,
    one_of_filter,
    FilterClause,
    FilterExpression,
    FilterField,
    FilterTerm,
    filter_expression_matches_document,
    stored_document_filter_index_has_logical_scope_postings,
)


def unique_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def test_build_stored_document_filter_index_captures_sorted_postings() raises:
    var stored = build_stored_document_filter_index(
        CollectionId("news"),
        SegmentId("segment-0001"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        [
            DocumentMetadataMap(
                [
                    DocumentMetadataEntry("source", "wire"),
                    DocumentMetadataEntry("language", "en"),
                ]
            ),
            DocumentMetadataMap(
                [DocumentMetadataEntry("source", "blog")]
            ),
            DocumentMetadataMap(
                [DocumentMetadataEntry("source", "wire")]
            ),
        ],
    )

    assert_equal(stored.document_count, 3)
    assert_equal(stored.posting_count(), 6)
    assert_equal(stored.doc_indices_for("source", "wire")[0], 0)
    assert_equal(stored.doc_indices_for("source", "wire")[1], 2)
    assert_equal(stored.doc_indices_for("language", "en")[0], 0)
    assert_equal(
        stored.doc_indices_for(FILTER_FIELD_INTERNAL_COLLECTION_ID, "news")[2],
        2,
    )
    assert_equal(
        stored.doc_indices_for(FILTER_FIELD_INTERNAL_TENANT_ID, "tenant-a")[1],
        1,
    )
    assert_equal(
        stored.doc_indices_for(FILTER_FIELD_INTERNAL_NAMESPACE_ID, "search")[0],
        0,
    )
    assert_equal(stored_document_filter_index_has_logical_scope_postings(stored), True)


def test_document_filter_index_store_roundtrip_preserves_postings() raises:
    var root = unique_root("kayak-document-filter-index")
    var stored = StoredDocumentFilterIndex(
        CollectionId("news"),
        SegmentId("segment-0001"),
        3,
        0,
        [
            DocumentFilterPosting("source", "wire", [0, 2]),
            DocumentFilterPosting("language", "en", [0]),
        ],
    )

    save_stored_document_filter_index(root, stored)
    var loaded = load_stored_document_filter_index(root)

    assert_equal(loaded.collection_id.value, "news")
    assert_equal(loaded.segment_id.value, "segment-0001")
    assert_equal(loaded.document_count, 3)
    assert_equal(loaded.posting_count(), 2)
    assert_equal(loaded.doc_indices_for("source", "wire")[1], 2)
    assert_equal(loaded.artifact_byte_size > 0, True)


def test_document_filter_allowlist_matches_exact_filter_runtime() raises:
    var doc_ids = List[String]()
    doc_ids.append("doc-a")
    doc_ids.append("doc-b")
    doc_ids.append("doc-c")
    var metadata_maps = [
        DocumentMetadataMap(
            [
                DocumentMetadataEntry("source", "wire"),
                DocumentMetadataEntry("language", "en"),
            ]
        ),
        DocumentMetadataMap(
            [DocumentMetadataEntry("source", "blog")]
        ),
        DocumentMetadataMap(),
    ]
    var stored = build_stored_document_filter_index(
        CollectionId("news"),
        SegmentId("segment-0001"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        metadata_maps,
    )
    var expression = FilterExpression(
        [
            FilterClause(
                [
                    FilterTerm(FilterField("source"), "eq", ["wire"]),
                    FilterTerm(FilterField("language"), "one_of", ["en", "fr"]),
                ]
            ),
            FilterClause(
                [FilterTerm(FilterField("doc_id"), "eq", ["doc-c"])]
            ),
        ]
    )
    var allowlist = document_filter_allowlist_for_expression(
        doc_ids,
        stored,
        expression,
    )

    assert_equal(allowlist.matching_document_count, 2)
    for doc_index in range(len(doc_ids)):
        assert_equal(
            allowlist.matches_document_index(doc_index),
            filter_expression_matches_document(
                expression,
                doc_ids[doc_index],
                metadata_maps[doc_index],
            ),
        )

    var match_all_allowlist = document_filter_allowlist_for_expression(
        doc_ids,
        stored,
        one_of_filter("doc_id", ["doc-a", "doc-c"]),
    )
    assert_equal(match_all_allowlist.matches_document_index(0), True)
    assert_equal(match_all_allowlist.matches_document_index(1), False)
    assert_equal(match_all_allowlist.matches_document_index(2), True)


def test_document_filter_allowlist_respects_internal_scope_postings() raises:
    var doc_ids = List[String]()
    doc_ids.append("doc-a")
    doc_ids.append("doc-b")
    doc_ids.append("doc-c")
    doc_ids.append("doc-d")
    var metadata_maps = [
        DocumentMetadataMap([DocumentMetadataEntry("source", "wire")]),
        DocumentMetadataMap([DocumentMetadataEntry("source", "wire")]),
        DocumentMetadataMap([DocumentMetadataEntry("source", "blog")]),
        DocumentMetadataMap([DocumentMetadataEntry("source", "wire")]),
    ]
    var stored = StoredDocumentFilterIndex(
        CollectionId("shared-pool"),
        SegmentId("segment-0001"),
        4,
        0,
        [
            DocumentFilterPosting(
                FILTER_FIELD_INTERNAL_COLLECTION_ID,
                "news",
                [0, 1, 2],
            ),
            DocumentFilterPosting(
                FILTER_FIELD_INTERNAL_COLLECTION_ID,
                "other",
                [3],
            ),
            DocumentFilterPosting(
                FILTER_FIELD_INTERNAL_TENANT_ID,
                "tenant-a",
                [0, 2],
            ),
            DocumentFilterPosting(
                FILTER_FIELD_INTERNAL_TENANT_ID,
                "tenant-b",
                [1, 3],
            ),
            DocumentFilterPosting(
                FILTER_FIELD_INTERNAL_NAMESPACE_ID,
                "search",
                [0, 1, 3],
            ),
            DocumentFilterPosting(
                FILTER_FIELD_INTERNAL_NAMESPACE_ID,
                "archive",
                [2],
            ),
            DocumentFilterPosting("source", "wire", [0, 1, 3]),
            DocumentFilterPosting("source", "blog", [2]),
        ],
    )
    var expression = conjoin_filter_expressions(
        one_of_filter("source", ["wire"]),
        logical_scope_filter(
            LogicalFilterScope("news", "tenant-a", "search")
        ),
    )
    var allowlist = document_filter_allowlist_for_expression(
        doc_ids,
        stored,
        expression,
    )
    var scopes = [
        LogicalFilterScope("news", "tenant-a", "search"),
        LogicalFilterScope("news", "tenant-b", "search"),
        LogicalFilterScope("news", "tenant-a", "archive"),
        LogicalFilterScope("other", "tenant-b", "search"),
    ]

    assert_equal(allowlist.matching_document_count, 1)
    for doc_index in range(len(doc_ids)):
        assert_equal(
            allowlist.matches_document_index(doc_index),
            filter_expression_matches_document_in_scope(
                expression,
                scopes[doc_index],
                doc_ids[doc_index],
                metadata_maps[doc_index],
            ),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
