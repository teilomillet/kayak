from std.testing import TestSuite, assert_equal

from kayak.filters import (
    FilterClause,
    FilterExpression,
    FilterField,
    FilterTerm,
    and_filter,
    match_all_filter,
    one_of_filter,
)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
