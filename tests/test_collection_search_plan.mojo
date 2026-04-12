from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    MetricScalar,
    VECTOR_SCALAR_NAME,
    pack_documents,
)
from kayak import (
    CollectionId,
    CollectionManifest,
    CollectionStats,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    TenantId,
    NamespaceId,
    StoredPackedIndex,
    collection_search_explain_json,
    exact_full_scan_search_plan,
    explain_collection_search,
    load_resolved_collection_snapshot,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_packed_index,
)


def write_segment(
    collection_root: Path,
    segment_id: String,
    generation: Int,
    documents: List[EncodedDocument],
) raises:
    var segment_root = collection_root / "segments" / segment_id
    var packed_index = pack_documents(documents)

    save_stored_packed_index(
        segment_root / "packed_index",
        StoredPackedIndex(
            "collection://search-plan",
            "colbertv2",
            VECTOR_SCALAR_NAME,
            packed_index.copy(),
        ),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId(segment_id),
            CollectionId("search-plan"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            generation,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "",
            SegmentStats(
                packed_index.document_count,
                packed_index.total_vector_count,
                packed_index.total_vector_count,
                512,
            ),
        ),
    )


def make_collection_root() raises -> Path:
    var root = Path("/tmp/kayak-collection-search-plan")
    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("search-plan"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            2,
        ),
    )

    write_segment(
        root,
        "segment-0001",
        1,
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0]]),
        ],
    )
    write_segment(
        root,
        "segment-0002",
        2,
        [
            EncodedDocument("doc-c", [[0.0, 1.0], [1.0, 0.0]]),
            EncodedDocument("doc-d", [[0.0, 1.0], [0.0, 1.0]]),
        ],
    )

    save_snapshot_manifest(
        root / "snapshots" / "snapshot-0002",
        SnapshotManifest(
            SnapshotId("snapshot-0002"),
            CollectionId("search-plan"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            2,
            [SegmentId("segment-0001"), SegmentId("segment-0002")],
            CollectionStats(2, 4, 8, 8, 1024),
        ),
    )
    return root^


def test_exact_full_scan_search_plan_explains_collection_snapshot() raises:
    var root = make_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0002"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var plan = exact_full_scan_search_plan(2, 3)
    var explain = explain_collection_search(
        ExactCpuBackend(), query, resolved, plan
    )
    var json = collection_search_explain_json(explain)

    assert_equal(explain.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(len(explain.candidate_set.hits), 3)
    assert_equal(len(explain.final_hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.final_hits[1].doc_id, "doc-c")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.candidate_stage.score_histogram.bin_count, 8)
    assert_equal(
        explain.exact_stage.score_histogram.counts[0]
            + explain.exact_stage.score_histogram.counts[1]
            + explain.exact_stage.score_histogram.counts[2]
            + explain.exact_stage.score_histogram.counts[3]
            + explain.exact_stage.score_histogram.counts[4]
            + explain.exact_stage.score_histogram.counts[5]
            + explain.exact_stage.score_histogram.counts[6]
            + explain.exact_stage.score_histogram.counts[7],
        len(explain.final_hits),
    )
    assert_equal(json.find("\"collection_id\":\"search-plan\"") != -1, True)
    assert_equal(json.find("\"candidate_generator_kind\":\"exact_full_scan\"") != -1, True)


def test_candidate_budget_rejects_candidate_k_below_final_k() raises:
    var raised = False

    try:
        _ = exact_full_scan_search_plan(3, 2)
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
