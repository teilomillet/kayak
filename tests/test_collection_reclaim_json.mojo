from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    CollectionReclaimExecutionResult,
    CollectionReclaimPlan,
    NamespaceId,
    SegmentId,
    SnapshotId,
    SnapshotRetentionDecision,
    TenantId,
    collection_reclaim_execution_result_json,
    collection_reclaim_plan_json,
)


def test_collection_reclaim_plan_json_contains_decisions_and_segments() raises:
    var json = collection_reclaim_plan_json(
        CollectionReclaimPlan(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "snapshot-0003",
            3,
            2,
            1,
            1,
            2,
            2048,
            [
                SnapshotRetentionDecision(
                    SnapshotId("snapshot-0003"),
                    3,
                    1,
                    1024,
                    True,
                    "active_snapshot",
                ),
                SnapshotRetentionDecision(
                    SnapshotId("snapshot-0002"),
                    2,
                    1,
                    1024,
                    True,
                    "retained_inactive_by_generation",
                ),
                SnapshotRetentionDecision(
                    SnapshotId("snapshot-0001"),
                    1,
                    1,
                    1024,
                    False,
                    "inactive_reclaim_candidate",
                ),
            ],
            [SegmentId("segment-1"), SegmentId("segment-2")],
        )
    )

    assert_equal(json.find("\"active_snapshot_id\":\"snapshot-0003\"") != -1, True)
    assert_equal(json.find("\"reclaimable_unique_segment_count\":2") != -1, True)
    assert_equal(json.find("\"reclaimable_unique_byte_size\":2048") != -1, True)
    assert_equal(json.find("\"snapshot_id\":\"snapshot-0001\"") != -1, True)
    assert_equal(json.find("\"reason\":\"inactive_reclaim_candidate\"") != -1, True)
    assert_equal(json.find("\"reclaimable_unique_segment_ids\":[\"segment-1\",\"segment-2\"]") != -1, True)


def test_collection_reclaim_execution_result_json_contains_apply_state() raises:
    var json = collection_reclaim_execution_result_json(
        CollectionReclaimExecutionResult(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            True,
            1,
            2,
            2048,
            [SnapshotId("snapshot-0001")],
            [SegmentId("segment-1"), SegmentId("segment-2")],
        )
    )

    assert_equal(json.find("\"applied\":true") != -1, True)
    assert_equal(json.find("\"snapshot_count\":1") != -1, True)
    assert_equal(json.find("\"unique_segment_count\":2") != -1, True)
    assert_equal(json.find("\"unique_byte_size\":2048") != -1, True)
    assert_equal(json.find("\"snapshot_ids\":[\"snapshot-0001\"]") != -1, True)
    assert_equal(json.find("\"unique_segment_ids\":[\"segment-1\",\"segment-2\"]") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
