# Query-time exact evaluation over the document-filter sidecar.
#
# This module owns clause/term evaluation against a segment-local allowlist
# index plus packed-index doc_ids. It does not own stage-1 execution or storage.

from std.collections import List

from kayak.filters import FilterExpression, FilterTerm

from .document_filter_index import StoredDocumentFilterIndex


struct DocumentFilterAllowlist(Copyable):
    var flags: List[Int]
    var matching_document_count: Int

    def __init__(
        out self,
        read flags: List[Int],
        matching_document_count: Int,
    ) raises:
        if matching_document_count < 0:
            raise Error(
                "document filter allowlist matching_document_count must be non-negative"
            )
        if matching_document_count > len(flags):
            raise Error(
                "document filter allowlist matching_document_count exceeds flags length"
            )
        self.flags = flags.copy()
        self.matching_document_count = matching_document_count

    def matches_document_index(self, document_index: Int) -> Bool:
        if document_index < 0 or document_index >= len(self.flags):
            return False

        return self.flags[document_index] != 0


def merge_sorted_doc_indices_union(
    read left: List[Int], read right: List[Int]
) -> List[Int]:
    var merged = List[Int]()
    var left_index = 0
    var right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            merged.append(left[left_index])
            left_index += 1
            right_index += 1
        elif left[left_index] < right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    while left_index < len(left):
        merged.append(left[left_index])
        left_index += 1

    while right_index < len(right):
        merged.append(right[right_index])
        right_index += 1

    return merged^


def intersect_sorted_doc_indices(
    read left: List[Int], read right: List[Int]
) -> List[Int]:
    var intersection = List[Int]()
    var left_index = 0
    var right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            intersection.append(left[left_index])
            left_index += 1
            right_index += 1
        elif left[left_index] < right[right_index]:
            left_index += 1
        else:
            right_index += 1

    return intersection^


def sorted_doc_indices_for_doc_id_term(
    read doc_ids: List[String], read term: FilterTerm
) -> List[Int]:
    var matches = List[Int]()

    for doc_index in range(len(doc_ids)):
        for value in term.values:
            if doc_ids[doc_index] == value:
                matches.append(doc_index)
                break

    return matches^


def sorted_doc_indices_for_metadata_term(
    read index: StoredDocumentFilterIndex, read term: FilterTerm
) -> List[Int]:
    var matches = List[Int]()

    for value in term.values:
        matches = merge_sorted_doc_indices_union(
            matches,
            index.doc_indices_for(term.field.name, value),
        )

    return matches^


def sorted_doc_indices_for_filter_term(
    read doc_ids: List[String],
    read index: StoredDocumentFilterIndex,
    read term: FilterTerm,
) -> List[Int]:
    if term.field.name == "doc_id":
        return sorted_doc_indices_for_doc_id_term(doc_ids, term)

    return sorted_doc_indices_for_metadata_term(index, term)


def sorted_doc_indices_for_filter_expression(
    read doc_ids: List[String],
    read index: StoredDocumentFilterIndex,
    read expression: FilterExpression,
) -> List[Int]:
    if expression.is_match_all():
        var matches = List[Int]()
        for doc_index in range(index.document_count):
            matches.append(doc_index)
        return matches^

    var expression_matches = List[Int]()

    for clause in expression.clauses:
        if len(clause.terms) == 0:
            continue

        var clause_matches = sorted_doc_indices_for_filter_term(
            doc_ids,
            index,
            clause.terms[0],
        )
        for term_index in range(1, len(clause.terms)):
            clause_matches = intersect_sorted_doc_indices(
                clause_matches,
                sorted_doc_indices_for_filter_term(
                    doc_ids,
                    index,
                    clause.terms[term_index],
                ),
            )
            if len(clause_matches) == 0:
                break

        expression_matches = merge_sorted_doc_indices_union(
            expression_matches,
            clause_matches,
        )
    return expression_matches^


def document_filter_allowlist_for_expression(
    read doc_ids: List[String],
    read index: StoredDocumentFilterIndex,
    read expression: FilterExpression,
) raises -> DocumentFilterAllowlist:
    if len(doc_ids) != index.document_count:
        raise Error(
            "document filter allowlist requires doc_ids aligned with filter index document_count"
        )

    var flags = List[Int]()
    for _ in range(index.document_count):
        flags.append(0)

    var matching_doc_indices = sorted_doc_indices_for_filter_expression(
        doc_ids,
        index,
        expression,
    )
    for doc_index in matching_doc_indices:
        flags[doc_index] = 1

    return DocumentFilterAllowlist(flags, len(matching_doc_indices))
