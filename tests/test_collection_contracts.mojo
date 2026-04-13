from std.testing import TestSuite, assert_equal

from kayak.collections import (
    COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
    COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED,
    CollectionId,
    CollectionManifest,
    CollectionStats,
    CompactionPlan,
    gem_graph_build_spec,
    NamespaceId,
    SearchArtifactBuildPolicy,
    SearchArtifactBuildSpec,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    StoredDocumentTextCorpus,
    TenantId,
    sealed_segment_has_text_corpus,
)
from kayak.numeric import MetricScalar, VECTOR_SCALAR_NAME
from kayak.text import DocumentTextCorpus


def test_segment_stats_keep_average_vectors_explicit() raises:
    var stats = SegmentStats(2, 9, 8, 1024)

    assert_equal(stats.document_count, 2)
    assert_equal(stats.token_count, 9)
    assert_equal(stats.total_vector_count, 8)
    assert_equal(stats.byte_size, 1024)
    assert_equal(stats.average_vectors_per_document, MetricScalar(4.0))


def test_collection_contracts_hold_serving_metadata() raises:
    var tenant_id = TenantId("tenant-a")
    var namespace_id = NamespaceId("search")
    var collection_id = CollectionId("news")
    var segment_id = SegmentId("segment-0001")
    var snapshot_id = SnapshotId("snapshot-0001")

    var collection = CollectionManifest(
        collection_id.copy(),
        tenant_id.copy(),
        namespace_id.copy(),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        128,
        3,
        "snapshot-0001",
        2,
        SearchArtifactBuildPolicy(
            [
                SearchArtifactBuildSpec("document_proxy", "proxy_sidecar"),
                SearchArtifactBuildSpec("centroid_postings", "postings_sidecar"),
            ]
        ),
    )
    var segment_stats = SegmentStats(2, 12, 10, 2048)
    var segment = SealedSegmentManifest(
        segment_id.copy(),
        collection_id.copy(),
        tenant_id.copy(),
        namespace_id.copy(),
        3,
        "colbertv2",
        VECTOR_SCALAR_NAME,
        128,
        "segments/segment-0001/packed_index",
        "",
        "segments/segment-0001/text_corpus",
        segment_stats.copy(),
    )
    var snapshot = SnapshotManifest(
        snapshot_id,
        collection_id.copy(),
        tenant_id.copy(),
        namespace_id.copy(),
        3,
        [segment_id.copy()],
        CollectionStats(1, 2, 12, 10, 2048),
    )
    var compaction = CompactionPlan(
        collection_id.copy(),
        tenant_id.copy(),
        namespace_id.copy(),
        [segment_id.copy()],
        SegmentId("segment-0002"),
        "merge small sealed segments",
        SegmentStats(2, 12, 10, 2048),
    )
    var text_corpus = StoredDocumentTextCorpus(
        collection_id,
        segment_id,
        DocumentTextCorpus(["doc-a", "doc-b"], ["alpha", "beta"]),
    )

    assert_equal(collection.model_name, "colbertv2")
    assert_equal(collection.latest_generation, 3)
    assert_equal(collection.active_snapshot_id, "snapshot-0001")
    assert_equal(
        collection.collection_layout_family,
        COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED,
    )
    assert_equal(collection.default_keep_latest_inactive_count, 2)
    assert_equal(
        collection.search_artifact_build_policy.stage1_artifacts[0].root,
        "proxy_sidecar",
    )
    assert_equal(segment.vector_dim, 128)
    assert_equal(sealed_segment_has_text_corpus(segment), True)
    assert_equal(len(snapshot.segment_ids), 1)
    assert_equal(compaction.reason, "merge small sealed segments")
    assert_equal(text_corpus.corpus.doc_ids[1], "doc-b")


def test_collection_ids_reject_empty_strings() raises:
    var raised = False

    try:
        _ = CollectionId("")
    except:
        raised = True

    assert_equal(raised, True)


def test_collection_manifest_rejects_invalid_stage1_build_config() raises:
    var raised = False

    try:
        _ = CollectionManifest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            0,
            SearchArtifactBuildPolicy(
                [SearchArtifactBuildSpec("gem_graph", "gem_graph")]
            ),
        )
    except:
        raised = True

    assert_equal(raised, True)


def test_collection_manifest_accepts_configured_gem_graph_build_spec() raises:
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        128,
        0,
        SearchArtifactBuildPolicy(
            [gem_graph_build_spec(2, 3, 1, "gem_graph", 4, 5)]
        ),
    )

    assert_equal(
        collection.search_artifact_build_policy.stage1_artifacts[0].family,
        "gem_graph",
    )


def test_collection_manifest_supports_explicit_shared_pool_layout() raises:
    var collection = CollectionManifest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        128,
        3,
        "snapshot-0003",
        1,
        SearchArtifactBuildPolicy(
            [SearchArtifactBuildSpec("document_proxy", "proxy_sidecar")]
        ),
        COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
    )

    assert_equal(
        collection.collection_layout_family,
        COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
