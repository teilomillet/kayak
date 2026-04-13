from std.testing import TestSuite, assert_equal

from kayak.filters import (
    FILTER_FIELD_INTERNAL_TENANT_ID,
    FilterClause,
    FilterExpression,
    FilterField,
    FilterTerm,
    and_filter,
    conjoin_filter_expressions,
    filter_expression_matches_document,
    filter_expression_matches_document_in_scope,
    filter_expression_is_exact_doc_id_filter,
    filter_expression_matches_doc_id,
    filter_expression_requires_document_metadata,
    logical_scope_filter,
    match_all_filter,
    one_of_filter,
    LogicalFilterScope,
)
from kayak import DocumentMetadataEntry, DocumentMetadataMap


def test_match_all_filter_has_zero_clauses() raises:
    var expression = match_all_filter()

    assert_equal(expression.is_match_all(), True)
    assert_equal(expression.clause_count(), 0)


def test_filter_expression_models_monotone_boolean_structure() raises:
    var expression = and_filter(
        [
            FilterTerm(FilterField("source"), "eq", ["wire"]),
            FilterTerm(FilterField("language"), "one_of", ["en", "fr"]),
        ]
    )
    var one_of = one_of_filter("author", ["alice", "bob"])

    assert_equal(expression.is_match_all(), False)
    assert_equal(expression.clause_count(), 1)
    assert_equal(expression.clauses[0].terms[0].field.name, "source")
    assert_equal(expression.clauses[0].terms[1].values[1], "fr")
    assert_equal(one_of.clauses[0].terms[0].operator, "one_of")


def test_filter_clause_rejects_empty_term_lists() raises:
    var raised = False

    try:
        _ = FilterClause([])
    except:
        raised = True

    assert_equal(raised, True)


def test_exact_doc_id_filter_runtime_support_is_explicit() raises:
    var doc_id_filter = one_of_filter("doc_id", ["doc-a", "doc-b"])
    var mixed_filter = and_filter(
        [FilterTerm(FilterField("source"), "eq", ["wire"])]
    )

    assert_equal(filter_expression_is_exact_doc_id_filter(doc_id_filter), True)
    assert_equal(filter_expression_matches_doc_id(doc_id_filter, "doc-a"), True)
    assert_equal(filter_expression_matches_doc_id(doc_id_filter, "doc-z"), False)
    assert_equal(filter_expression_is_exact_doc_id_filter(mixed_filter), False)


def test_filter_expression_matches_document_metadata_exactly() raises:
    var metadata_filter = and_filter(
        [FilterTerm(FilterField("source"), "eq", ["wire"])]
    )
    var metadata = DocumentMetadataMap(
        [DocumentMetadataEntry("source", "wire")]
    )

    assert_equal(
        filter_expression_requires_document_metadata(metadata_filter), True
    )
    assert_equal(
        filter_expression_matches_document(
            metadata_filter,
            "doc-a",
            metadata,
        ),
        True,
    )
    assert_equal(
        filter_expression_matches_document(
            metadata_filter,
            "doc-a",
            DocumentMetadataMap(),
        ),
        False,
    )


def test_conjoined_logical_scope_filters_match_only_the_scoped_documents() raises:
    var scoped_expression = conjoin_filter_expressions(
        one_of_filter("source", ["wire"]),
        logical_scope_filter(
            LogicalFilterScope("news", "tenant-a", "search")
        ),
    )
    var metadata = DocumentMetadataMap(
        [DocumentMetadataEntry("source", "wire")]
    )

    assert_equal(
        filter_expression_requires_document_metadata(scoped_expression),
        True,
    )
    assert_equal(
        filter_expression_matches_document_in_scope(
            scoped_expression,
            LogicalFilterScope("news", "tenant-a", "search"),
            "doc-a",
            metadata,
        ),
        True,
    )
    assert_equal(
        filter_expression_matches_document_in_scope(
            scoped_expression,
            LogicalFilterScope("news", "tenant-b", "search"),
            "doc-a",
            metadata,
        ),
        False,
    )
    assert_equal(
        filter_expression_matches_document_in_scope(
            scoped_expression,
            LogicalFilterScope("news", "tenant-a", "archive"),
            "doc-a",
            metadata,
        ),
        False,
    )
    assert_equal(
        scoped_expression.clauses[0].terms[2].field.name,
        FILTER_FIELD_INTERNAL_TENANT_ID,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
