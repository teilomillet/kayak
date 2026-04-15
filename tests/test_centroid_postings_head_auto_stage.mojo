from std.testing import TestSuite, assert_equal

from kayak.planning.centroid_postings_head_auto_stage import (
    auto_centroid_head_base_posting_cap,
)


def test_head_auto_shrinks_long_query_tiny_window_regime() raises:
    assert_equal(auto_centroid_head_base_posting_cap(10, 32, 128), 4)
    assert_equal(auto_centroid_head_base_posting_cap(3, 32, 128), 3)


def test_head_auto_keeps_existing_low_query_expansion_regimes() raises:
    assert_equal(auto_centroid_head_base_posting_cap(20, 8, 128), 20)
    assert_equal(auto_centroid_head_base_posting_cap(40, 16, 128), 32)


def test_head_auto_keeps_default_cap_outside_narrow_shrink_window() raises:
    assert_equal(auto_centroid_head_base_posting_cap(20, 32, 128), 8)
    assert_equal(auto_centroid_head_base_posting_cap(40, 32, 128), 4)
    assert_equal(auto_centroid_head_base_posting_cap(80, 32, 128), 16)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
