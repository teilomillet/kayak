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
    StoredDocumentTextCorpus,
    SegmentId,
    SegmentStats,
    SealedSegmentManifest,
    SnapshotId,
    SnapshotManifest,
    TenantId,
    NamespaceId,
    StoredPackedIndex,
    best_effort_faithfulness_policy,
    build_stored_centroid_heads_index,
    build_stored_centroid_posting_index,
    build_stored_document_proxy_index,
    centroid_heads_search_plan,
    centroid_posting_flat_scores_for_segment,
    centroid_posting_imputed_flat_scores_for_segment,
    centroid_posting_imputed_scores_for_segment,
    centroid_postings_head_auto_search_plan,
    centroid_posting_scores_for_segment,
    centroid_postings_blockmax_search_plan,
    centroid_postings_flat_search_plan,
    centroid_postings_head_search_plan,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_search_plan,
    collection_search_explain_json,
    document_proxy_search_plan,
    exact_full_scan_clause_text_search_plan,
    exact_full_scan_search_plan,
    explain_collection_search,
    gem_graph_search_artifact,
    gem_graph_search_plan,
    loaded_segment_stored_centroid_postings_index,
    load_resolved_collection_snapshot,
    oracle_full_recall_required_faithfulness_policy,
    build_stored_gem_graph_index,
    save_collection_manifest,
    save_stored_centroid_heads_index,
    save_stored_centroid_posting_index,
    save_stored_document_text_corpus,
    save_sealed_segment_manifest,
    save_snapshot_manifest,
    save_stored_document_proxy_index,
    save_stored_gem_graph_index,
    save_stored_packed_index,
)
from kayak.text import DocumentTextCorpus


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


def make_clause_text_collection_root() raises -> Path:
    var root = Path("/tmp/kayak-collection-clause-text")
    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("search-plan"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            1,
        ),
    )

    var segment_root = root / "segments" / "segment-0001"
    var documents = [
        EncodedDocument("doc-context", [[1.0, 0.0], [1.0, 0.0]]),
        EncodedDocument("doc-answer", [[1.0, 0.0], [0.8, 0.2]]),
    ]
    var packed_index = pack_documents(documents)

    save_stored_packed_index(
        segment_root / "packed_index",
        StoredPackedIndex(
            "collection://clause-text",
            "colbertv2",
            VECTOR_SCALAR_NAME,
            packed_index.copy(),
        ),
    )
    save_stored_document_text_corpus(
        segment_root / "text_corpus",
        StoredDocumentTextCorpus(
            CollectionId("search-plan"),
            SegmentId("segment-0001"),
            DocumentTextCorpus(
                ["doc-context", "doc-answer"],
                [
                    "Gugulethu township logo emblem heritage schools history",
                    "Zama Dance School was founded in 1984 in a church and the longest serving employee is the artistic director.",
                ],
            ),
        ),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("search-plan"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "",
            "",
            "",
            "text_corpus",
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
            CollectionId("search-plan"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 2, 4, 4, 512),
        ),
    )
    return root^


def make_document_proxy_collection_root() raises -> Path:
    var root = Path("/tmp/kayak-collection-document-proxy")
    var segment_root = root / "segments" / "segment-0001"
    var packed_index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[0.6, 0.6], [0.6, 0.6]]),
        ]
    )
    var stored_index = StoredPackedIndex(
        "collection://document-proxy",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        packed_index.copy(),
    )

    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("document-proxy"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            1,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())
    save_stored_document_proxy_index(
        segment_root / "document_proxy",
        build_stored_document_proxy_index(stored_index, 0),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("document-proxy"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "document_proxy",
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
            CollectionId("document-proxy"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 2, 4, 4, 512),
        ),
    )
    return root^


def make_centroid_postings_collection_root() raises -> Path:
    var root = Path("/tmp/kayak-collection-centroid-postings")
    var segment_root = root / "segments" / "segment-0001"
    var packed_index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[0.6, 0.6], [0.6, 0.6]]),
        ]
    )
    var stored_index = StoredPackedIndex(
        "collection://centroid-postings",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        packed_index.copy(),
    )

    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("centroid-postings"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            1,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())
    save_stored_centroid_posting_index(
        segment_root / "centroid_postings",
        build_stored_centroid_posting_index(stored_index, 0),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("centroid-postings"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "centroid_postings",
            "",
            "",
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
            CollectionId("centroid-postings"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 2, 4, 4, 512),
        ),
    )
    return root^


def make_gem_graph_collection_root() raises -> Path:
    var root = Path("/tmp/kayak-collection-gem-graph")
    var segment_root = root / "segments" / "segment-0001"
    var packed_index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0]]),
            EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
        ]
    )
    var stored_index = StoredPackedIndex(
        "collection://gem-graph",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        packed_index.copy(),
    )

    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("gem-graph"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            1,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())
    save_stored_gem_graph_index(
        segment_root / "gem_graph",
        build_stored_gem_graph_index(stored_index, 2, 2, 2, 2, 2),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("gem-graph"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            [gem_graph_search_artifact("gem_graph")],
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
            CollectionId("gem-graph"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 3, 6, 6, 512),
        ),
    )
    return root^


def make_centroid_heads_collection_root() raises -> Path:
    var root = Path("/tmp/kayak-collection-centroid-heads")
    var segment_root = root / "segments" / "segment-0001"
    var packed_index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[0.6, 0.6], [0.6, 0.6]]),
        ]
    )
    var stored_index = StoredPackedIndex(
        "collection://centroid-heads",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        packed_index.copy(),
    )

    save_collection_manifest(
        root,
        CollectionManifest(
            CollectionId("centroid-heads"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            1,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())
    save_stored_centroid_heads_index(
        segment_root / "centroid_heads",
        build_stored_centroid_heads_index(stored_index, 0, 1),
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            SegmentId("segment-0001"),
            CollectionId("centroid-heads"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            "packed_index",
            "",
            "centroid_heads",
            "",
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
            CollectionId("centroid-heads"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            1,
            [SegmentId("segment-0001")],
            CollectionStats(1, 2, 4, 4, 512),
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
    assert_equal(explain.exact_stage.stage_name, "exact_oracle")
    assert_equal(explain.exact_stage.document_count, 4)
    assert_equal(explain.exact_stage.output_hit_count, 2)
    assert_equal(explain.exact_stage.vector_count, 8)
    assert_equal(explain.exact_stage.byte_size > 0, True)
    assert_equal(explain.candidate_stage.score_histogram.bin_count, 8)
    assert_equal(
        explain.stage2.score_histogram.counts[0]
            + explain.stage2.score_histogram.counts[1]
            + explain.stage2.score_histogram.counts[2]
            + explain.stage2.score_histogram.counts[3]
            + explain.stage2.score_histogram.counts[4]
            + explain.stage2.score_histogram.counts[5]
            + explain.stage2.score_histogram.counts[6]
            + explain.stage2.score_histogram.counts[7],
        len(explain.final_hits),
    )
    assert_equal(json.find("\"collection_id\":\"search-plan\"") != -1, True)
    assert_equal(json.find("\"candidate_generator_kind\":\"exact_full_scan\"") != -1, True)
    assert_equal(json.find("\"candidate_generator_family\":\"exact\"") != -1, True)
    assert_equal(json.find("\"faithfulness_policy_kind\":\"exact_stage1_required\"") != -1, True)
    assert_equal(json.find("\"faithfulness\":") != -1, True)
    assert_equal(json.find("\"stage1_required_artifact_families\":[]") != -1, True)
    assert_equal(json.find("\"stage2_kind\":\"noop_topk\"") != -1, True)
    assert_equal(json.find("\"stage2_family\":\"identity\"") != -1, True)


def test_exact_full_scan_clause_text_stage2_can_refine_exact_candidates() raises:
    var root = make_clause_text_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [1.0, 0.0]])
    var plan = exact_full_scan_clause_text_search_plan(1, 2)
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        plan,
        query_text=
            "Gugulethu township logo. founded in 1984 in a church longest serving employee artistic director",
    )
    var json = collection_search_explain_json(explain)

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-context")
    assert_equal(explain.final_hits[0].doc_id, "doc-answer")
    assert_equal(explain.plan.stage2_operator.kind, "clause_text")
    assert_equal(explain.plan.stage2_operator.family, "text")
    assert_equal(explain.plan.stage2_operator.requires_query_text, True)
    assert_equal(explain.stage2.stage_name, "clause_text")
    assert_equal(explain.stage2.document_count, 2)
    assert_equal(explain.stage2.token_count > 0, True)
    assert_equal(explain.stage2.byte_size > 0, True)
    assert_equal(json.find("\"stage2_kind\":\"clause_text\"") != -1, True)
    assert_equal(json.find("\"stage2_requires_query_text\":true") != -1, True)


def test_candidate_budget_rejects_candidate_k_below_final_k() raises:
    var raised = False

    try:
        _ = exact_full_scan_search_plan(3, 2)
    except:
        raised = True

    assert_equal(raised, True)


def test_document_proxy_search_plan_exact_reranks_shortlist() raises:
    var root = make_document_proxy_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        document_proxy_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "document_proxy")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")
    assert_equal(explain.candidate_stage.vector_count, 2)
    assert_equal(explain.stage2.document_count, 2)


def test_document_proxy_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_document_proxy_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        document_proxy_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "centroid_postings")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")
    assert_equal(explain.candidate_stage.vector_count, 2)
    assert_equal(explain.candidate_stage.token_count, 3)
    assert_equal(explain.stage2.document_count, 2)


def test_centroid_postings_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_flat_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_flat_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "centroid_postings_flat")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")


def test_centroid_postings_flat_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_flat_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_flat_stage_matches_plain_stage_scores() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var index = loaded_segment_stored_centroid_postings_index(
        resolved.segments[0]
    ).index.copy()
    var plain_scores = centroid_posting_scores_for_segment(query.token_vectors, index)
    var flat_scores = centroid_posting_flat_scores_for_segment(query, index)

    assert_equal(len(flat_scores), len(plain_scores))
    for score_index in range(len(plain_scores)):
        assert_equal(flat_scores[score_index], plain_scores[score_index])


def test_centroid_heads_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_heads_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_heads_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "centroid_heads")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")
    assert_equal(explain.candidate_stage.vector_count, 2)
    assert_equal(explain.candidate_stage.token_count, 2)


def test_centroid_heads_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_centroid_heads_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_heads_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_head_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_head_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "centroid_postings_head")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")


def test_centroid_postings_head_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_head_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_head_auto_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_head_auto_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "centroid_postings_head_auto")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")


def test_centroid_postings_head_auto_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_head_auto_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_head_requires_weight_sorted_sidecar() raises:
    var root = make_centroid_postings_collection_root()
    var manifest_path = (
        root / "segments" / "segment-0001" / "centroid_postings" / "manifest.tsv"
    )
    manifest_path.write_text(
        manifest_path.read_text().replace("posting_order_kind\tweight_desc_doc_asc\n", "")
    )
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var raised = False

    try:
        _ = explain_collection_search(
            ExactCpuBackend(),
            EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
            resolved,
            centroid_postings_head_search_plan(
                1, 2, oracle_full_recall_required_faithfulness_policy()
            ),
        )
    except:
        raised = True

    assert_equal(raised, True)


def test_centroid_postings_head_auto_requires_weight_sorted_sidecar() raises:
    var root = make_centroid_postings_collection_root()
    var manifest_path = (
        root / "segments" / "segment-0001" / "centroid_postings" / "manifest.tsv"
    )
    manifest_path.write_text(
        manifest_path.read_text().replace("posting_order_kind\tweight_desc_doc_asc\n", "")
    )
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var raised = False

    try:
        _ = explain_collection_search(
            ExactCpuBackend(),
            EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
            resolved,
            centroid_postings_head_auto_search_plan(
                1, 2, oracle_full_recall_required_faithfulness_policy()
            ),
        )
    except:
        raised = True

    assert_equal(raised, True)


def test_centroid_postings_blockmax_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_blockmax_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "centroid_postings_blockmax")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")


def test_centroid_postings_blockmax_search_plan_keeps_exact_winner_on_toy_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_blockmax_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-a")
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")


def test_centroid_postings_blockmax_requires_weight_sorted_sidecar() raises:
    var root = make_centroid_postings_collection_root()
    var manifest_path = (
        root / "segments" / "segment-0001" / "centroid_postings" / "manifest.tsv"
    )
    manifest_path.write_text(
        manifest_path.read_text().replace("posting_order_kind\tweight_desc_doc_asc\n", "")
    )
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var raised = False

    try:
        _ = explain_collection_search(
            ExactCpuBackend(),
            EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
            resolved,
            centroid_postings_blockmax_search_plan(
                1, 2, oracle_full_recall_required_faithfulness_policy()
            ),
        )
    except:
        raised = True

    assert_equal(raised, True)


def test_centroid_postings_imputed_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_imputed_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.plan.candidate_generator.kind, "centroid_postings_imputed")
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")


def test_centroid_postings_imputed_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_imputed_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_imputed_flat_search_plan_exact_reranks_shortlist() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_imputed_flat_search_plan(
            1, 2, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(
        explain.plan.candidate_generator.kind, "centroid_postings_imputed_flat"
    )
    assert_equal(len(explain.candidate_set.hits), 2)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))
    assert_equal(explain.faithfulness.passes, True)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_full_recall")


def test_centroid_postings_imputed_flat_search_plan_reports_oracle_miss_when_shortlist_is_too_small() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        centroid_postings_imputed_flat_search_plan(
            1, 1, oracle_full_recall_required_faithfulness_policy()
        ),
    )

    assert_equal(explain.candidate_set.hits[0].doc_id, "doc-b")
    assert_equal(explain.final_hits[0].doc_id, "doc-b")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(0.0))
    assert_equal(explain.faithfulness.passes, False)
    assert_equal(explain.faithfulness.evidence_kind, "oracle_recall_loss")


def test_centroid_postings_imputed_flat_stage_matches_imputed_stage_scores() raises:
    var root = make_centroid_postings_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var index = loaded_segment_stored_centroid_postings_index(
        resolved.segments[0]
    ).index.copy()
    var imputed_scores = centroid_posting_imputed_scores_for_segment(
        query.token_vectors,
        index,
        1,
    )
    var imputed_flat_scores = centroid_posting_imputed_flat_scores_for_segment(
        query,
        index,
        1,
    )

    assert_equal(len(imputed_flat_scores), len(imputed_scores))
    for score_index in range(len(imputed_scores)):
        assert_equal(imputed_flat_scores[score_index], imputed_scores[score_index])


def test_gem_graph_search_plan_executes_with_exact_rerank() raises:
    var root = make_gem_graph_collection_root()
    var resolved = load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var explain = explain_collection_search(
        ExactCpuBackend(),
        query,
        resolved,
        gem_graph_search_plan(
            1,
            2,
            oracle_full_recall_required_faithfulness_policy(),
            1,
            4,
        ),
    )
    var json = collection_search_explain_json(explain)

    assert_equal(len(explain.candidate_set.hits) > 0, True)
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(
        explain.candidate_set.graph_search_counters.visited_vertex_count > 0,
        True,
    )
    assert_equal(
        explain.candidate_stage.graph_search_counters.expanded_edge_count > 0,
        True,
    )
    assert_equal(
        explain.candidate_stage.graph_search_counters.visited_cluster_count > 0,
        True,
    )
    assert_equal(
        explain.candidate_stage.graph_search_counters.entry_point_count > 0,
        True,
    )
    assert_equal(
        explain.candidate_stage.graph_search_counters.max_frontier_size > 0,
        True,
    )
    assert_equal(json.find("\"graph_search_counters\":") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
