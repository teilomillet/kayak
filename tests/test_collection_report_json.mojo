from std.testing import TestSuite, assert_equal

from kayak.collections import (
    CollectionStats,
    CollectionStorageReport,
    SegmentStats,
    SegmentStorageReport,
    collection_storage_report_json,
)


def test_collection_storage_report_json_contains_density_and_segments() raises:
    var json = collection_storage_report_json(
        CollectionStorageReport(
            CollectionStats(1, 4, 8, 8, 1024),
            1,
            0,
            [
                SegmentStorageReport(
                    "segment-0001",
                    True,
                    SegmentStats(4, 8, 8, 1024),
                )
            ],
        )
    )

    assert_equal(json.find("\"bytes_per_document\":256.0") != -1, True)
    assert_equal(json.find("\"segment_id\":\"segment-0001\"") != -1, True)
    assert_equal(json.find("\"segment_count_with_text\":1") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
