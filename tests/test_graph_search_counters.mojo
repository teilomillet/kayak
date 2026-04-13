from std.testing import TestSuite, assert_equal

from kayak import (
    GraphSearchCounters,
    accumulate_graph_search_counters,
    graph_search_counters_have_activity,
)


def test_graph_search_counters_have_activity_requires_non_zero_signal() raises:
    assert_equal(graph_search_counters_have_activity(GraphSearchCounters()), False)
    assert_equal(
        graph_search_counters_have_activity(GraphSearchCounters(0, 0, 1, 0, 0)),
        True,
    )
    assert_equal(
        graph_search_counters_have_activity(GraphSearchCounters(0, 0, 0, 0, 3)),
        True,
    )


def test_accumulate_graph_search_counters_sums_counts_and_keeps_peak_frontier() raises:
    var total = accumulate_graph_search_counters(
        GraphSearchCounters(2, 5, 1, 1, 4),
        GraphSearchCounters(3, 7, 2, 4, 9),
    )

    assert_equal(total.visited_vertex_count, 5)
    assert_equal(total.expanded_edge_count, 12)
    assert_equal(total.visited_cluster_count, 3)
    assert_equal(total.entry_point_count, 5)
    assert_equal(total.max_frontier_size, 9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
