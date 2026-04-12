from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionHit,
    ScoreScalar,
    build_score_histogram,
    empty_score_histogram,
)


def test_empty_score_histogram_uses_zero_bins() raises:
    var histogram = empty_score_histogram()

    assert_equal(histogram.bin_count, 0)
    assert_equal(len(histogram.counts), 0)


def test_score_histogram_counts_hits_across_bins() raises:
    var histogram = build_score_histogram(
        [
            CollectionHit("segment-0001", "doc-a", ScoreScalar(1.0)),
            CollectionHit("segment-0001", "doc-b", ScoreScalar(2.0)),
            CollectionHit("segment-0001", "doc-c", ScoreScalar(3.0)),
            CollectionHit("segment-0001", "doc-d", ScoreScalar(4.0)),
        ],
        4,
    )

    assert_equal(histogram.bin_count, 4)
    assert_equal(histogram.min_score, 1.0)
    assert_equal(histogram.max_score, 4.0)
    assert_equal(histogram.counts[0], 1)
    assert_equal(histogram.counts[3], 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
