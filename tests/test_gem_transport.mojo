from std.testing import TestSuite, assert_equal

from kayak.index.gem_transport import min_cost_transport_distance


def test_min_cost_transport_distance_matches_simple_exact_plan() raises:
    var distance = min_cost_transport_distance([0.5, 0.5], [1.0], [0.0, 1.0])
    assert_equal(distance, 0.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
