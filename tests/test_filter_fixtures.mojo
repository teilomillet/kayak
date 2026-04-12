from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    default_filter_selectivity_fixtures,
    high_selectivity_filter_fixture,
    low_selectivity_filter_fixture,
)
from kayak.numeric import MetricScalar


def test_filter_selectivity_fixtures_cover_low_and_high_selectivity() raises:
    var low = low_selectivity_filter_fixture()
    var high = high_selectivity_filter_fixture()
    var fixtures = default_filter_selectivity_fixtures()

    assert_equal(low.layout_name, "shared_pool")
    assert_equal(low.selectivity(), MetricScalar(0.02))
    assert_equal(high.layout_name, "tenant_isolated")
    assert_equal(high.selectivity(), MetricScalar(0.8))
    assert_equal(len(fixtures), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
