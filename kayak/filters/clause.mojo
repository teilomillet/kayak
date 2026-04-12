from std.collections import List

from .term import FilterTerm


struct FilterClause(Copyable):
    var terms: List[FilterTerm]

    def __init__(out self, read terms: List[FilterTerm]) raises:
        if len(terms) == 0:
            raise Error("filter clause must contain at least one term")

        self.terms = terms.copy()
