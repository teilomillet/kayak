from .expression import FilterExpression


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


def filter_expression_matches_doc_id(
    read expression: FilterExpression, doc_id: String
) -> Bool:
    if expression.is_match_all():
        return True

    for clause in expression.clauses:
        var clause_matches = True
        for term in clause.terms:
            var term_matches = False
            for value in term.values:
                if value == doc_id:
                    term_matches = True
                    break

            if not term_matches:
                clause_matches = False
                break

        if clause_matches:
            return True

    return False
