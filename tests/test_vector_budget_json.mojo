from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    VectorBudgetSweepSummary,
    standard_document_vector_budget_sizes,
    standard_query_vector_budget_sizes,
    vector_budget_sweep_summaries_json,
)


def test_standard_query_vector_budget_sizes_cap_and_deduplicate() raises:
    var sizes = standard_query_vector_budget_sizes(10)

    assert_equal(len(sizes), 3)
    assert_equal(sizes[0], 4)
    assert_equal(sizes[1], 8)
    assert_equal(sizes[2], 10)


def test_standard_document_vector_budget_sizes_cap_and_deduplicate() raises:
    var sizes = standard_document_vector_budget_sizes(24)

    assert_equal(len(sizes), 3)
    assert_equal(sizes[0], 8)
    assert_equal(sizes[1], 16)
    assert_equal(sizes[2], 24)


def test_vector_budget_json_contains_budget_fields() raises:
    var json = vector_budget_sweep_summaries_json(
        [
            VectorBudgetSweepSummary(
                "mock://dataset",
                "mock_collection",
                "snapshot-0001",
                "mock-model",
                "document_proxy",
                10,
                40,
                8,
                32,
                4,
                0.75,
                0.60,
                0.70,
                0.80,
                1.0,
            )
        ]
    )

    assert_equal(json.find("\"query_vector_budget\":8") != -1, True)
    assert_equal(json.find("\"document_vector_budget\":32") != -1, True)
    assert_equal(
        json.find("\"candidate_generator_kind\":\"document_proxy\"") != -1,
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
