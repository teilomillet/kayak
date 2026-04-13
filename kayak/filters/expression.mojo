from std.collections import List

from .clause import FilterClause
from .field import FilterField
from .term import FilterTerm


struct FilterExpression(Copyable):
    var clauses: List[FilterClause]

    def __init__(out self, read clauses: List[FilterClause]):
        self.clauses = clauses.copy()

    def is_match_all(self) -> Bool:
        return len(self.clauses) == 0

    def clause_count(self) -> Int:
        return len(self.clauses)


def match_all_filter() -> FilterExpression:
    return FilterExpression([])


def and_filter(read terms: List[FilterTerm]) raises -> FilterExpression:
    return FilterExpression([FilterClause(terms)])


def one_of_filter(field_name: String, read values: List[String]) raises -> FilterExpression:
    return and_filter(
        [FilterTerm(FilterField(field_name), "one_of", values)]
    )


def conjoin_filter_expressions(
    read left: FilterExpression, read right: FilterExpression
) raises -> FilterExpression:
    if left.is_match_all():
        return right.copy()
    if right.is_match_all():
        return left.copy()

    var clauses = List[FilterClause]()
    for left_clause in left.clauses:
        for right_clause in right.clauses:
            var terms = List[FilterTerm]()
            for term in left_clause.terms:
                terms.append(term.copy())
            for term in right_clause.terms:
                terms.append(term.copy())
            clauses.append(FilterClause(terms))

    return FilterExpression(clauses)
