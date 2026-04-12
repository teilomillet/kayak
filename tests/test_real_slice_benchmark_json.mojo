from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    RealSliceBenchmarkSummary,
    real_slice_benchmark_summaries_json,
)


def test_real_slice_benchmark_summary_json_contains_machine_readable_fields() raises:
    var json = real_slice_benchmark_summaries_json(
        [
            RealSliceBenchmarkSummary(
                "mock://dataset",
                "mock-model",
                "mock",
                "json",
                "storage",
                "storage",
                "mrr",
                1.0,
                0.9,
                1.0,
                1.0,
                1.0,
                0.0123,
                10,
                2,
                8,
                3,
                32,
                128,
            )
        ]
    )

    assert_equal(json.find("\"dataset_id\":\"mock://dataset\"") != -1, True)
    assert_equal(json.find("\"mean_search_seconds\":0.0123") != -1, True)
    assert_equal(
        json.find("\"nominal_document_vector_count\":32") != -1, True
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
