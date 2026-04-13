from std.testing import TestSuite, assert_equal

from kayak import (
    SEARCH_PLANNER_STATUS_EXACT_FALLBACK,
    SEARCH_PLANNER_STATUS_EXPERIMENTAL,
    SEARCH_PLANNER_STATUS_PROMOTED,
    default_candidate_generator_order_for_goal,
    registered_search_planner_candidate_generator_kinds,
    search_planner_registry_entry,
)


def test_default_candidate_generator_orders_are_registry_driven() raises:
    var balanced = default_candidate_generator_order_for_goal("balanced")
    var latency_first = default_candidate_generator_order_for_goal("latency_first")
    var native = default_candidate_generator_order_for_goal("native_multivector")

    assert_equal(balanced[0], "centroid_postings_imputed_flat")
    assert_equal(latency_first[0], "document_proxy")
    assert_equal(native[0], "centroid_postings_imputed_flat")
    assert_equal(balanced[len(balanced) - 1], "exact_full_scan")


def test_registry_marks_current_promoted_and_experimental_generators() raises:
    assert_equal(
        search_planner_registry_entry("document_proxy").planner_status,
        SEARCH_PLANNER_STATUS_PROMOTED,
    )
    assert_equal(
        search_planner_registry_entry("gem_graph").planner_status,
        SEARCH_PLANNER_STATUS_EXPERIMENTAL,
    )
    assert_equal(
        search_planner_registry_entry("exact_full_scan").planner_status,
        SEARCH_PLANNER_STATUS_EXACT_FALLBACK,
    )
    assert_equal(
        registered_search_planner_candidate_generator_kinds()[0],
        "exact_full_scan",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
