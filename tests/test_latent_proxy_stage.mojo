from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CandidateBudget,
    CandidateGenerator,
    CollectionId,
    CollectionManifest,
    CollectionStats,
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    MetricScalar,
    NamespaceId,
    SearchPlan,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    TenantId,
    VECTOR_SCALAR_NAME,
    exact_late_interaction_reference_scoring_semantics,
    exact_late_interaction_stage2_reference_operator,
    explain_collection_search,
    load_resolved_collection_snapshot,
    none_stage3_verifier_operator,
    oracle_full_recall_required_faithfulness_policy,
    pack_documents,
    save_collection_manifest,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
)
from kayak.collections import (
    latent_proxy_search_artifact,
    loaded_segment_has_latent_proxy_index,
)
from kayak.index import (
    LATENT_PROXY_ACTIVATION_RELU,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
    LatentProxyIndex,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
)
from kayak.storage import (
    StoredLatentProxyIndex,
    StoredPackedIndex,
    save_stored_latent_proxy_index,
    save_stored_packed_index,
)


def make_latent_proxy_collection_root() raises -> Path:
    var root = Path("/tmp/kayak-collection-latent-proxy")
    var segment_root = root / "segments" / "segment-0001"
    var packed_index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
            EncodedDocument("doc-b", [[0.0, 1.0], [0.0, 1.0]]),
        ]
    )
    var stored_index = StoredPackedIndex(
        "collection://latent-proxy",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        packed_index.copy(),
    )

    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("latent-proxy"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            1,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())
    save_stored_latent_proxy_index(
        segment_root / "latent_proxy",
        StoredLatentProxyIndex(
            "collection://latent-proxy",
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            0,
            LatentQueryProjection(
                2,
                2,
                1.0,
                [
                    LatentQueryProjectionBlock(
                        LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
                        LATENT_PROXY_ACTIVATION_RELU,
                        2,
                        2,
                        [[1.0, 0.0], [0.0, 1.0]],
                        [0.0, 0.0],
                        1.0,
                        False,
                        0.00001,
                        [],
                        [],
                    )
                ],
            ),
            LatentProxyIndex(
                ["doc-a", "doc-b"],
                [[1.0, -1.0], [-1.0, 1.0]],
                2,
            ),
        ),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("latent-proxy"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            [latent_proxy_search_artifact("latent_proxy")],
            "",
            SegmentStats(
                packed_index.document_count,
                packed_index.total_vector_count,
                packed_index.total_vector_count,
                512,
            ),
        ),
    )
    save_snapshot_manifest(
        root / "snapshots" / "snapshot-0001",
        SnapshotManifest(
            SnapshotId("snapshot-0001"),
            CollectionId("latent-proxy"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 2, 4, 4, 512),
        ),
    )
    return root^


def test_latent_proxy_candidates_shortlist_then_exact_rerank() raises:
    var root = make_latent_proxy_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [1.0, 0.0]])
    var plan = SearchPlan(
        CandidateGenerator("latent_proxy"),
        CandidateBudget(1, 2),
        oracle_full_recall_required_faithfulness_policy(),
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        plan,
    )

    assert_equal(loaded_segment_has_latent_proxy_index(resolved.segments[0]), True)
    assert_equal(explain.plan.candidate_generator.kind, "latent_proxy")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-a")
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.candidate_stage.vector_count, 2)
    assert_equal(explain.stage2.document_count, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
