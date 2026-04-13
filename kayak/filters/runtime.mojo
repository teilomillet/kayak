from kayak.collections.document_metadata import (
    DocumentMetadataMap,
    empty_document_metadata_map,
)

from .expression import FilterExpression
from .logical_scope import (
    LogicalFilterScope,
    filter_field_is_internal_logical_scope,
    logical_scope_value_for_filter_field,
    unscoped_logical_filter_scope,
)


def filter_expression_is_exact_doc_id_filter(read expression: FilterExpression) -> Bool:
    if expression.is_match_all():
        return True

    if expression.clause_count() == 0:
        return True

    for clause in expression.clauses:
        if len(clause.terms) == 0:
            return False

        for term in clause.terms:
            if term.field.name != "doc_id":
                return False
            if term.operator != "eq" and term.operator != "one_of":
                return False

    return True


def filter_expression_requires_document_metadata(
    read expression: FilterExpression
) -> Bool:
    if expression.is_match_all():
        return False

    for clause in expression.clauses:
        for term in clause.terms:
            if term.field.name != "doc_id":
                return True

    return False


def filter_expression_matches_document(
    read expression: FilterExpression,
    doc_id: String,
    read metadata: DocumentMetadataMap,
) -> Bool:
    return filter_expression_matches_document_in_scope(
        expression,
        unscoped_logical_filter_scope(),
        doc_id,
        metadata,
    )


def filter_expression_matches_document_in_scope(
    read expression: FilterExpression,
    read scope: LogicalFilterScope,
    doc_id: String,
    read metadata: DocumentMetadataMap,
) -> Bool:
    if expression.is_match_all():
        return True

    for clause in expression.clauses:
        var clause_matches = True
        for term in clause.terms:
            var actual_value = String()
            if term.field.name == "doc_id":
                actual_value = doc_id.copy()
            elif filter_field_is_internal_logical_scope(term.field.name):
                actual_value = logical_scope_value_for_filter_field(
                    scope,
                    term.field.name,
                )
            else:
                for entry in metadata.entries:
                    if entry.key == term.field.name:
                        actual_value = entry.value.copy()
                        break

            var term_matches = False
            for value in term.values:
                if value == actual_value:
                    term_matches = True
                    break

            if not term_matches:
                clause_matches = False
                break

        if clause_matches:
            return True

    return False


def filter_expression_matches_doc_id(
    read expression: FilterExpression, doc_id: String
) -> Bool:
    return filter_expression_matches_document(
        expression,
        doc_id,
        empty_document_metadata_map(),
    )
