from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import EncodedDocument, VECTOR_SCALAR_NAME, pack_documents
from kayak.collections import (
    CollectionId,
    CollectionManifest,
    CollectionStats,
    NamespaceId,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    TenantId,
    document_proxy_search_artifact,
    centroid_postings_search_artifact,
    load_snapshot_search_artifact_availability,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
)
from kayak.planning import (
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
    select_search_plan_for_availability,
)
from kayak.storage import StoredPackedIndex, save_stored_packed_index


def write_segment_payload(
    segment_root: Path,
    collection_id: String,
    model_name: String,
    documents: List[EncodedDocument],
) raises:
    save_stored_packed_index(
        segment_root / "packed_index",
        StoredPackedIndex(
            collection_id,
            model_name,
            VECTOR_SCALAR_NAME,
            pack_documents(documents),
        ),
    )


def write_mixed_sidecar_snapshot_fixture() raises -> Path:
    var collection_root = Path("/tmp/kayak-snapshot-inventory-any-all")
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            2,
        ),
    )

    var segment_one_root = collection_root / "segments" / "segment-0001"
    var segment_two_root = collection_root / "segments" / "segment-0002"
    write_segment_payload(
        segment_one_root,
        "collection://news",
        "colbertv2",
        [EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]])],
    )
    write_segment_payload(
        segment_two_root,
        "collection://news",
        "colbertv2",
        [EncodedDocument("doc-b", [[0.0, 1.0], [1.0, 0.0]])],
    )
    save_sealed_segment_manifest(
        segment_one_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            [
                document_proxy_search_artifact("document_proxy"),
                centroid_postings_search_artifact("centroid_postings"),
            ],
            "",
            SegmentStats(1, 2, 2, 512),
        ),
    )
    save_sealed_segment_manifest(
        segment_two_root,
        SealedSegmentManifest(
            SegmentId("segment-0002"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            2,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            [document_proxy_search_artifact("document_proxy")],
            "",
            SegmentStats(1, 2, 2, 512),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / "snapshot-0002",
        SnapshotManifest(
            SnapshotId("snapshot-0002"),
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            2,
            [SegmentId("segment-0001"), SegmentId("segment-0002")],
            CollectionStats(2, 2, 4, 4, 1024),
        ),
    )

    return collection_root


def test_snapshot_inventory_reports_any_and_all_segment_families() raises:
    var collection_root = write_mixed_sidecar_snapshot_fixture()

    var availability = load_snapshot_search_artifact_availability(
        collection_root,
        SnapshotId("snapshot-0002"),
    )

    assert_equal(availability.segment_count, 2)
    assert_equal(
        availability.has_search_artifact_family_on_all_segments("document_proxy"),
        True,
    )
    assert_equal(
        availability.has_search_artifact_family_on_all_segments("centroid_postings"),
        False,
    )
    assert_equal(
        availability.search_artifact_families_available_on_any_segment[0],
        "document_proxy",
    )
    assert_equal(
        availability.search_artifact_families_available_on_any_segment[1],
        "centroid_postings",
    )


def test_planner_skips_partial_sidecar_family_on_multi_segment_snapshot() raises:
    var collection_root = write_mixed_sidecar_snapshot_fixture()
    var availability = load_snapshot_search_artifact_availability(
        collection_root,
        SnapshotId("snapshot-0002"),
    )
    var selection = select_search_plan_for_availability(
        availability,
        SearchPlanSelectionRequest(
            1,
            10,
            best_effort_faithfulness_policy(),
            goal=SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
        ),
    )

    assert_equal(selection.plan.candidate_generator.kind, "document_proxy")
    assert_equal(selection.selected_candidate_generator_status, "promoted")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
