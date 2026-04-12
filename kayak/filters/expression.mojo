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
