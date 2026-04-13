from std.testing import TestSuite, assert_equal
from std.pathlib import Path

from kayak import (
    CollectionId,
    CollectionManifest,
    CollectionStats,
    NamespaceId,
    SealedSegmentManifest,
    SegmentId,
    SnapshotId,
    SnapshotManifest,
    TenantId,
    VECTOR_SCALAR_NAME,
    build_compaction_plan_for_snapshot,
    execute_compaction_plan,
    load_collection_manifest,
    load_resolved_collection_snapshot,
    load_snapshot_manifest,
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


def aggregate_stats(
    read left: SealedSegmentManifest,
    read right: SealedSegmentManifest,
) raises -> CollectionStats:
    return CollectionStats(
        2,
        left.stats.document_count + right.stats.document_count,
        left.stats.token_count + right.stats.token_count,
        left.stats.total_vector_count + right.stats.total_vector_count,
        left.stats.byte_size + right.stats.byte_size,
    )


def test_execute_compaction_plan_publishes_replacement_snapshot() raises:
    var root = unique_root("kayak-collection-compaction")
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        2,
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
    save_snapshot_manifest(
        root / "snapshots" / "snapshot-0002",
        SnapshotManifest(
            SnapshotId("snapshot-0002"),
            collection.collection_id,
            collection.tenant_id,
            collection.namespace_id,
            2,
            [segment_one.segment_id.copy(), segment_two.segment_id.copy()],
            aggregate_stats(segment_one, segment_two),
        ),
    )

    var plan = build_compaction_plan_for_snapshot(
        root,
        SnapshotId("snapshot-0002"),
        [SegmentId("segment-1"), SegmentId("segment-2")],
        SegmentId("segment-3"),
        "merge visible segments",
    )
    var replacement = execute_compaction_plan(
        root,
        SnapshotId("snapshot-0002"),
        SnapshotId("snapshot-0003"),
        plan,
    )
    var published_collection = load_collection_manifest(root)
    var preserved_snapshot = load_snapshot_manifest(root / "snapshots" / "snapshot-0002")
    var published_snapshot = load_snapshot_manifest(root / "snapshots" / "snapshot-0003")
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0003"))

    assert_equal(plan.expected_output_stats.document_count, 2)
    assert_equal(plan.expected_output_stats.token_count, 4)
    assert_equal(plan.expected_output_stats.total_vector_count, 4)
    assert_equal(published_collection.latest_generation, 3)
    assert_equal(replacement.generation, 3)
    assert_equal(len(preserved_snapshot.segment_ids), 2)
    assert_equal(len(published_snapshot.segment_ids), 1)
    assert_equal(published_snapshot.segment_ids[0].value, "segment-3")
    assert_equal(resolved.snapshot.stats.document_count, 2)
    assert_equal(resolved.snapshot.stats.segment_count, 1)
    assert_equal(resolved.segments[0].stored_index.index.doc_ids[0], "doc-a")
    assert_equal(resolved.segments[0].stored_index.index.doc_ids[1], "doc-b")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
