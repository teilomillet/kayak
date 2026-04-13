from std.testing import TestSuite, assert_equal

from kayak import CandidateGenerator
from kayak.benchmarks.posting_cap_json import (
    PostingCapSweepSummary,
    posting_cap_sweep_summaries_json,
    standard_posting_cap_sizes,
)


def test_standard_posting_cap_sizes_cap_and_deduplicate() raises:
    var sizes = standard_posting_cap_sizes(10)

    assert_equal(len(sizes), 5)
    assert_equal(sizes[0], 1)
    assert_equal(sizes[1], 2)
    assert_equal(sizes[2], 4)
    assert_equal(sizes[3], 8)
    assert_equal(sizes[4], 10)


def test_posting_cap_json_contains_axis_and_stage1_fields() raises:
    var json = posting_cap_sweep_summaries_json(
        [
            PostingCapSweepSummary(
                "mock://dataset",
                "mock_collection",
                "snapshot-0001",
                "mock-model",
                CandidateGenerator("centroid_heads"),
                10,
                40,
                8,
                32,
                16,
                4,
                0.0125,
                0.95,
                0.60,
                0.70,
                0.80,
                1.0,
                32,
                128,
                4096,
            )
        ]
    )

    assert_equal(json.find("\"posting_cap\":16") != -1, True)
    assert_equal(json.find("\"stage1_byte_size\":4096") != -1, True)
    assert_equal(json.find("\"stage1_token_count\":128") != -1, True)
    assert_equal(
        json.find("\"candidate_generator_kind\":\"centroid_heads\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"candidate_generator_family\":\"centroid\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"stage1_interaction_semantics\":\"approximate_late_interaction\"")
            != -1,
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
