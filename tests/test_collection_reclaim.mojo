from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    CollectionManifest,
    CollectionStats,
    NamespaceId,
    SealedSegmentManifest,
    SegmentId,
    SnapshotId,
    SnapshotManifest,
    SnapshotRetentionPolicy,
    TenantId,
    VECTOR_SCALAR_NAME,
    build_collection_reclaim_plan,
    save_collection_manifest,
    save_snapshot_manifest,
    seal_single_segment,
)
from kayak.contracts import EncodedDocument


def unique_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def snapshot_stats_for_single_segment(
    read segment: SealedSegmentManifest
) raises -> CollectionStats:
    return CollectionStats(
        1,
        segment.stats.document_count,
        segment.stats.token_count,
        segment.stats.total_vector_count,
        segment.stats.byte_size,
    )


def build_three_snapshot_fixture(root: Path) raises -> CollectionManifest:
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        3,
        "snapshot-0003",
    )
    save_collection_manifest(root, collection)

    var segment_one = seal_single_segment(
        root,
        collection,
        SegmentId("segment-1"),
        1,
        [EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]])],
        ["alpha"],
    )
    var segment_two = seal_single_segment(
        root,
        collection,
        SegmentId("segment-2"),
        2,
        [EncodedDocument("doc-b", [[0.0, 1.0], [1.0, 0.0]])],
        ["beta"],
    )
    var segment_three = seal_single_segment(
        root,
        collection,
        SegmentId("segment-3"),
        3,
        [EncodedDocument("doc-c", [[1.0, 0.0], [1.0, 0.0]])],
        ["gamma"],
    )

    save_snapshot_manifest(
        root / "snapshots" / "snapshot-0001",
        SnapshotManifest(
            SnapshotId("snapshot-0001"),
            collection.collection_id,
            collection.tenant_id,
            collection.namespace_id,
            1,
            [segment_one.segment_id.copy()],
            snapshot_stats_for_single_segment(segment_one),
        ),
    )
    save_snapshot_manifest(
        root / "snapshots" / "snapshot-0002",
        SnapshotManifest(
            SnapshotId("snapshot-0002"),
            collection.collection_id,
            collection.tenant_id,
            collection.namespace_id,
            2,
            [segment_two.segment_id.copy()],
            snapshot_stats_for_single_segment(segment_two),
        ),
    )
    save_snapshot_manifest(
        root / "snapshots" / "snapshot-0003",
        SnapshotManifest(
            SnapshotId("snapshot-0003"),
            collection.collection_id,
            collection.tenant_id,
            collection.namespace_id,
            3,
            [segment_three.segment_id.copy()],
            snapshot_stats_for_single_segment(segment_three),
        ),
    )

    _ = segment_one
    _ = segment_two
    _ = segment_three
    return collection^


def test_collection_reclaim_plan_keeps_latest_inactive_snapshot() raises:
    var root = unique_root("kayak-collection-reclaim-latest")
    _ = build_three_snapshot_fixture(root)

    var plan = build_collection_reclaim_plan(root, SnapshotRetentionPolicy(1))

    assert_equal(plan.active_snapshot_id, "snapshot-0003")
    assert_equal(plan.total_snapshot_count, 3)
    assert_equal(plan.inactive_snapshot_count, 2)
    assert_equal(plan.retained_inactive_snapshot_count, 1)
    assert_equal(plan.reclaimable_snapshot_count, 1)
    assert_equal(plan.reclaimable_unique_segment_count, 1)
    assert_equal(plan.reclaimable_unique_segment_ids[0].value, "segment-1")
    assert_equal(plan.reclaimable_unique_byte_size > 0, True)
    assert_equal(plan.decisions[0].snapshot_id.value, "snapshot-0003")
    assert_equal(plan.decisions[0].reason, "active_snapshot")
    assert_equal(plan.decisions[1].snapshot_id.value, "snapshot-0002")
    assert_equal(plan.decisions[1].reason, "retained_inactive_by_generation")
    assert_equal(plan.decisions[2].snapshot_id.value, "snapshot-0001")
    assert_equal(plan.decisions[2].reason, "inactive_reclaim_candidate")


def test_collection_reclaim_plan_respects_pinned_snapshots() raises:
    var root = unique_root("kayak-collection-reclaim-pinned")
    _ = build_three_snapshot_fixture(root)

    var plan = build_collection_reclaim_plan(
        root,
        SnapshotRetentionPolicy(0, [SnapshotId("snapshot-0001")]),
    )

    assert_equal(plan.retained_inactive_snapshot_count, 1)
    assert_equal(plan.reclaimable_snapshot_count, 1)
    assert_equal(plan.reclaimable_unique_segment_count, 1)
    assert_equal(plan.reclaimable_unique_segment_ids[0].value, "segment-2")
    assert_equal(plan.decisions[1].snapshot_id.value, "snapshot-0002")
    assert_equal(plan.decisions[1].reason, "inactive_reclaim_candidate")
    assert_equal(plan.decisions[2].snapshot_id.value, "snapshot-0001")
    assert_equal(plan.decisions[2].reason, "pinned_snapshot")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
