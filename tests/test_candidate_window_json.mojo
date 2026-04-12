from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    CandidateWindowSweepSummary,
    candidate_window_sweep_summaries_json,
    standard_candidate_window_sizes,
)


def test_standard_candidate_window_sizes_deduplicates_and_caps_document_count() raises:
    var sizes = standard_candidate_window_sizes(10, 30)

    assert_equal(len(sizes), 3)
    assert_equal(sizes[0], 10)
    assert_equal(sizes[1], 20)
    assert_equal(sizes[2], 30)


def test_candidate_window_sweep_json_contains_recall_fields() raises:
    var json = candidate_window_sweep_summaries_json(
        [
            CandidateWindowSweepSummary(
                "mock://dataset",
                "mock_collection",
                "snapshot-0001",
                "mock-model",
                10,
                20,
                4,
                20.0,
                0.95,
            )
        ]
    )

    assert_equal(
        json.find("\"mean_candidate_recall_at_final_k\":0.95") != -1, True
    )
    assert_equal(json.find("\"candidate_k\":20") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
