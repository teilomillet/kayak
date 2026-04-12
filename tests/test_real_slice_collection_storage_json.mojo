from std.testing import TestSuite, assert_equal

from kayak.benchmarks import (
    RealSliceCollectionStorageSummary,
    real_slice_collection_storage_summaries_json,
)


def test_real_slice_collection_storage_json_contains_density_fields() raises:
    var json = real_slice_collection_storage_summaries_json(
        [
            RealSliceCollectionStorageSummary(
                "mock://dataset",
                "mock_collection",
                "mock-model",
                "snapshot-0001",
                1,
                10,
                20,
                20,
                4096,
                409.6,
                204.8,
                204.8,
                0,
                1,
            )
        ]
    )

    assert_equal(json.find("\"dataset_id\":\"mock://dataset\"") != -1, True)
    assert_equal(json.find("\"bytes_per_document\":409.6") != -1, True)
    assert_equal(json.find("\"segment_count_without_text\":1") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
