from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    BuildReclaimPlanRequest,
    CollectionId,
    CollectionLifecycleRequest,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    DeleteDocumentsRequest,
    DocumentMetadataUpdate,
    EncodedDocument,
    EncodedQuery,
    ExecuteReclaimRequest,
    ExactCpuBackend,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
    NamespaceId,
    PlannedSearchRequest,
    SearchPlanSelectionRequest,
    SearchArtifactBuildPolicy,
    SnapshotId,
    SnapshotRetentionPolicy,
    TenantId,
    UpdateCollectionRetentionPolicyRequest,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    best_effort_faithfulness_policy,
    build_collection_lifecycle_report,
    build_reclaim_plan,
    build_service_health_status,
    build_service_metrics_snapshot,
    create_collection,
    create_snapshot,
    default_exact_search_request,
    delete_documents,
    document_proxy_search_plan,
    execute_debug_search,
    execute_reclaim,
    execute_planned_debug_search,
    execute_planned_search,
    execute_search,
    exact_full_scan_clause_text_search_plan,
    exact_full_scan_search_plan,
    exact_late_interaction_clause_text_stage2_operator,
    export_snapshot,
    gem_graph_build_spec,
    gem_graph_search_plan,
    import_snapshot,
    load_collection_manifest,
    match_all_filter,
    one_of_filter,
    SearchRequest,
    update_collection_retention_policy,
    upsert_documents,
)


def unique_service_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def make_query() raises -> EncodedQuery:
    return EncodedQuery([[1.0, 0.0], [0.0, 1.0]])


def make_document(
    doc_id: String, read vectors: List[List[Float32]]
) raises -> EncodedDocument:
    return EncodedDocument(doc_id, vectors.copy())


def build_three_snapshot_service_fixture(
    service_root: Path,
    default_keep_latest_inactive_count: Int = 1,
) raises -> Path:
    var collection_root = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            default_keep_latest_inactive_count,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]), "alpha")],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish first snapshot",
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]), "beta")],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0002"),
            "publish second snapshot",
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-c", [[1.0, 0.0], [1.0, 0.0]]), "gamma")],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0003"),
            "publish third snapshot",
        ),
    )
    return collection_root


def test_hosted_collection_runtime_supports_mutate_snapshot_search_and_import() raises:
    var service_root = unique_service_root("kayak-service-runtime")
    var import_service_root = unique_service_root("kayak-service-runtime-import")

    var create_request = CreateCollectionRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
    )
    var collection_root = create_collection(service_root, create_request)

    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[0.0, 1.0], [0.0, 1.0]]),
                    "alpha draft",
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [0.0, 1.0]]),
                    "beta",
                ),
            ],
        ),
    )
    var document_count_after_upsert = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha revised",
                )
            ],
        ),
    )
    var document_count_after_delete = delete_documents(
        service_root,
        DeleteDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            ["doc-b"],
        ),
    )

    var snapshot = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "seal revised draft",
        ),
    )
    var search_request = default_exact_search_request(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        make_query(),
        1,
        True,
    )
    var search_response = execute_search(ExactCpuBackend(), service_root, search_request)
    var debug_response = execute_debug_search(
        ExactCpuBackend(), service_root, search_request
    )

    var bundle_root = unique_service_root("kayak-service-runtime-bundle")
    var exported = export_snapshot(
        service_root,
        ExportSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
        ),
        bundle_root,
    )
    var imported = import_snapshot(
        import_service_root,
        ImportSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "file://" + String(bundle_root),
        ),
    )
    var imported_search = execute_search(
        ExactCpuBackend(), import_service_root, search_request
    )
    var collection_manifest = load_collection_manifest(collection_root)

    assert_equal(document_count_after_upsert, 2)
    assert_equal(document_count_after_delete, 1)
    assert_equal(snapshot.generation, 1)
    assert_equal(collection_manifest.latest_generation, 1)
    assert_equal(collection_manifest.active_snapshot_id, "snapshot-0001")
    assert_equal(search_response.hits[0].doc_id, "doc-a")
    assert_equal(debug_response.explain.final_hits[0].doc_id, "doc-a")
    assert_equal(debug_response.explain.snapshot_id, "snapshot-0001")
    assert_equal(exported.snapshot_id.value, "snapshot-0001")
    assert_equal(imported.snapshot_id.value, "snapshot-0001")
    assert_equal(imported_search.hits[0].doc_id, "doc-a")

def test_hosted_collection_runtime_supports_text_family_stage2() raises:
    var service_root = unique_service_root("kayak-service-runtime-clause-text")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-context", [[1.0, 0.0], [1.0, 0.0]]),
                    "Gugulethu township logo emblem heritage schools history",
                ),
                UpsertDocument(
                    make_document("doc-answer", [[1.0, 0.0], [0.8, 0.2]]),
                    "Zama Dance School was founded in 1984 in a church and the longest serving employee is the artistic director.",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish clause text fixture",
        ),
    )

    var request = SearchRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        EncodedQuery([[1.0, 0.0], [1.0, 0.0]]),
        "Gugulethu township logo. founded in 1984 in a church longest serving employee artistic director",
        one_of_filter("doc_id", ["doc-context", "doc-answer"]),
        exact_full_scan_clause_text_search_plan(1, 2),
        True,
    )
    var response = execute_search(ExactCpuBackend(), service_root, request)
    var debug = execute_debug_search(ExactCpuBackend(), service_root, request)

    assert_equal(response.hits[0].doc_id, "doc-answer")
    assert_equal(debug.explain.candidate_set.hits[0].doc_id, "doc-context")
    assert_equal(debug.explain.final_hits[0].doc_id, "doc-answer")
    assert_equal(debug.explain.stage2.stage_name, "noop_topk")
    assert_equal(debug.explain.stage3_verifier.stage_name, "clause_text")
    assert_equal(debug.explain.stage3_verifier.token_count > 0, True)


def test_hosted_collection_runtime_supports_hybrid_stage2() raises:
    var service_root = unique_service_root("kayak-service-runtime-hybrid-stage2")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-context", [[1.0, 0.0], [1.0, 0.0]]),
                    "Gugulethu township logo emblem heritage schools history",
                ),
                UpsertDocument(
                    make_document("doc-answer", [[1.0, 0.0], [0.0, 1.0]]),
                    "Zama Dance School was founded in 1984 in a church and the longest serving employee is the artistic director.",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish hybrid stage2 fixture",
        ),
    )

    var request = SearchRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
        "Gugulethu township logo. founded in 1984 in a church longest serving employee artistic director",
        match_all_filter(),
        document_proxy_search_plan(
            1,
            2,
            best_effort_faithfulness_policy(),
            exact_late_interaction_clause_text_stage2_operator(),
        ),
        True,
    )
    var response = execute_search(ExactCpuBackend(), service_root, request)
    var debug = execute_debug_search(ExactCpuBackend(), service_root, request)

    assert_equal(
        response.plan.stage2_operator.kind,
        "exact_late_interaction_clause_text",
    )
    assert_equal(response.hits[0].doc_id, "doc-answer")
    assert_equal(debug.explain.candidate_set.hits[0].doc_id, "doc-context")
    assert_equal(debug.explain.final_hits[0].doc_id, "doc-answer")
    assert_equal(debug.explain.stage2.stage_name, "exact_late_interaction")
    assert_equal(debug.explain.stage3_verifier.stage_name, "clause_text")
    assert_equal(len(debug.explain.stage2.materialized_artifacts), 1)
    assert_equal(len(debug.explain.stage3_verifier.materialized_artifacts), 1)


def test_hosted_collection_runtime_supports_planned_search_after_import() raises:
    var service_root = unique_service_root("kayak-service-runtime-planned-import")
    var import_service_root = unique_service_root(
        "kayak-service-runtime-planned-import-target"
    )

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
                    "beta",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish import planner fixture",
        ),
    )
    var bundle_root = unique_service_root("kayak-service-runtime-planned-bundle")
    _ = export_snapshot(
        service_root,
        ExportSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
        ),
        bundle_root,
    )
    _ = import_snapshot(
        import_service_root,
        ImportSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "file://" + String(bundle_root),
        ),
    )

    var response = execute_planned_search(
        ExactCpuBackend(),
        import_service_root,
        PlannedSearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            match_all_filter(),
            SearchPlanSelectionRequest(
                1,
                2,
                best_effort_faithfulness_policy(),
                goal="balanced",
            ),
        ),
    )

    assert_equal(
        response.selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(response.search.hits[0].doc_id, "doc-a")


def test_hosted_collection_runtime_compacts_draft_after_snapshot() raises:
    var service_root = unique_service_root("kayak-service-runtime-draft-compact")

    var create_request = CreateCollectionRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
    )
    var collection_root = create_collection(service_root, create_request)

    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]), "alpha")],
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]), "beta")],
        ),
    )
    var document_count_after_delete = delete_documents(
        service_root,
        DeleteDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            ["doc-a"],
        ),
    )
    var snapshot = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "seal current draft",
        ),
    )

    var draft_manifest = (collection_root / "draft" / "manifest.tsv").read_text()
    assert_equal(draft_manifest.find("storage_kind\tmutation_log") >= 0, True)
    assert_equal(draft_manifest.find("mutation_count\t0") >= 0, True)
    assert_equal(
        (collection_root / "draft" / "mutations").exists(),
        False,
    )
    assert_equal(
        (collection_root / "draft" / "packed_index" / "manifest.tsv").exists(),
        True,
    )
    assert_equal(
        (collection_root / "draft" / "text_corpus" / "manifest.tsv").exists(),
        True,
    )
    assert_equal(
        load_collection_manifest(collection_root).active_snapshot_id,
        "snapshot-0001",
    )
    assert_equal(
        (collection_root / "draft" / "document_metadata" / "manifest.tsv").exists(),
        True,
    )
    assert_equal(document_count_after_delete, 1)
    assert_equal(snapshot.stats.document_count, 1)
    assert_equal(snapshot.segment_ids[0].value, "segment-1")

    var post_compaction_count = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-c", [[1.0, 0.0], [1.0, 0.0]]), "gamma")],
        ),
    )
    var snapshot_two = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0002"),
            "seal compacted baseline plus one mutation",
        ),
    )

    assert_equal(post_compaction_count, 2)
    assert_equal(snapshot_two.stats.document_count, 2)


def test_hosted_service_metrics_aggregate_visible_snapshots() raises:
    var service_root = unique_service_root("kayak-service-runtime-metrics")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("blogs"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )

    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]), "alpha")],
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("blogs"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]), "beta")],
        ),
    )

    var news_snapshot = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish news",
        ),
    )
    var blogs_snapshot = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("blogs"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish blogs",
        ),
    )

    var health = build_service_health_status(service_root)
    var metrics = build_service_metrics_snapshot(service_root)

    assert_equal(health.status, "ok")
    assert_equal(health.collection_count, 2)
    assert_equal(health.live_snapshot_count, 2)
    assert_equal(health.pending_compaction_count, 0)
    assert_equal(metrics.collection_count, 2)
    assert_equal(
        metrics.segment_count,
        news_snapshot.stats.segment_count + blogs_snapshot.stats.segment_count,
    )
    assert_equal(
        metrics.document_count,
        news_snapshot.stats.document_count + blogs_snapshot.stats.document_count,
    )
    assert_equal(
        metrics.vector_count,
        news_snapshot.stats.total_vector_count + blogs_snapshot.stats.total_vector_count,
    )
    assert_equal(
        metrics.byte_size,
        news_snapshot.stats.byte_size + blogs_snapshot.stats.byte_size,
    )
    assert_equal(metrics.published_snapshot_count, 2)
    assert_equal(metrics.inactive_snapshot_count, 0)
    assert_equal(metrics.inactive_unique_segment_count, 0)
    assert_equal(metrics.inactive_unique_byte_size, 0)
    assert_equal(metrics.pending_draft_collection_count, 0)
    assert_equal(metrics.pending_draft_mutation_count, 0)


def test_hosted_service_metrics_track_inactive_snapshots_and_pending_drafts() raises:
    var service_root = unique_service_root("kayak-service-runtime-operational")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]), "alpha")],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish first snapshot",
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]), "beta")],
        ),
    )
    var active_snapshot = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0002"),
            "publish second snapshot",
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-c", [[1.0, 0.0], [1.0, 0.0]]), "gamma")],
        ),
    )

    var metrics = build_service_metrics_snapshot(service_root)

    assert_equal(metrics.collection_count, 1)
    assert_equal(metrics.segment_count, active_snapshot.stats.segment_count)
    assert_equal(metrics.document_count, active_snapshot.stats.document_count)
    assert_equal(
        metrics.vector_count, active_snapshot.stats.total_vector_count
    )
    assert_equal(metrics.byte_size, active_snapshot.stats.byte_size)
    assert_equal(metrics.published_snapshot_count, 2)
    assert_equal(metrics.inactive_snapshot_count, 1)
    assert_equal(metrics.inactive_unique_segment_count, 1)
    assert_equal(metrics.inactive_unique_byte_size > 0, True)
    assert_equal(metrics.pending_draft_collection_count, 1)
    assert_equal(metrics.pending_draft_mutation_count, 1)


def test_hosted_collection_runtime_supports_exact_doc_id_filters() raises:
    var service_root = unique_service_root("kayak-service-runtime-doc-id-filter")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[0.0, 1.0], [0.0, 1.0]]),
                    "alpha",
                ),
                UpsertDocument(
                    make_document("doc-b", [[1.0, 0.0], [0.0, 1.0]]),
                    "beta",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish filter fixture",
        ),
    )

    var exact_filtered = execute_search(
        ExactCpuBackend(),
        service_root,
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            one_of_filter("doc_id", ["doc-a"]),
            exact_full_scan_search_plan(1, 1),
            False,
        ),
    )

    var proxy_filtered = execute_search(
        ExactCpuBackend(),
        service_root,
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            one_of_filter("doc_id", ["doc-a"]),
            document_proxy_search_plan(
                1, 1, best_effort_faithfulness_policy()
            ),
            False,
        ),
    )

    assert_equal(exact_filtered.hits[0].doc_id, "doc-a")
    assert_equal(proxy_filtered.hits[0].doc_id, "doc-a")


def test_hosted_collection_runtime_executes_planned_search_with_balanced_goal() raises:
    var service_root = unique_service_root("kayak-service-runtime-planned-balanced")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
                    "beta",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish planned balanced fixture",
        ),
    )

    var response = execute_planned_search(
        ExactCpuBackend(),
        service_root,
        PlannedSearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            match_all_filter(),
            SearchPlanSelectionRequest(
                1,
                2,
                best_effort_faithfulness_policy(),
            ),
        ),
    )

    assert_equal(
        response.selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(
        response.search.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(response.search.hits[0].doc_id, "doc-a")


def test_hosted_collection_runtime_keeps_native_stage1_for_exact_doc_id_filters() raises:
    var service_root = unique_service_root(
        "kayak-service-runtime-planned-doc-id-filter"
    )

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
                    "beta",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish planned doc-id fixture",
        ),
    )

    var response = execute_planned_search(
        ExactCpuBackend(),
        service_root,
        PlannedSearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            one_of_filter("doc_id", ["doc-b"]),
            SearchPlanSelectionRequest(
                1,
                2,
                best_effort_faithfulness_policy(),
            ),
        ),
    )

    assert_equal(
        response.selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(
        response.search.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(response.search.hits[0].doc_id, "doc-b")


def test_hosted_collection_runtime_executes_planned_search_with_clause_text_stage2() raises:
    var service_root = unique_service_root(
        "kayak-service-runtime-planned-clause-text"
    )

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-context", [[1.0, 0.0], [1.0, 0.0]]),
                    "Gugulethu township logo emblem heritage schools history",
                ),
                UpsertDocument(
                    make_document("doc-answer", [[1.0, 0.0], [0.8, 0.2]]),
                    "Zama Dance School was founded in 1984 in a church and the longest serving employee is the artistic director.",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish planned clause-text fixture",
        ),
    )

    var response = execute_planned_search(
        ExactCpuBackend(),
        service_root,
        PlannedSearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            EncodedQuery([[1.0, 0.0], [1.0, 0.0]]),
            "Gugulethu township logo. founded in 1984 in a church longest serving employee artistic director",
            "clause_text",
            one_of_filter("doc_id", ["doc-context", "doc-answer"]),
            SearchPlanSelectionRequest(
                1,
                2,
                best_effort_faithfulness_policy(),
                one_of_filter("doc_id", ["doc-context", "doc-answer"]),
                "exact_only",
                [],
                True,
            ),
        ),
    )

    assert_equal(response.selection.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(response.search.plan.stage2_operator.kind, "clause_text")
    assert_equal(response.search.hits[0].doc_id, "doc-answer")


def test_hosted_collection_runtime_executes_planned_search_with_hybrid_stage2() raises:
    var service_root = unique_service_root(
        "kayak-service-runtime-planned-hybrid-stage2"
    )

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-context", [[1.0, 0.0], [1.0, 0.0]]),
                    "Gugulethu township logo emblem heritage schools history",
                ),
                UpsertDocument(
                    make_document("doc-answer", [[1.0, 0.0], [0.0, 1.0]]),
                    "Zama Dance School was founded in 1984 in a church and the longest serving employee is the artistic director.",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish planned hybrid fixture",
        ),
    )

    var response = execute_planned_search(
        ExactCpuBackend(),
        service_root,
        PlannedSearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
            "Gugulethu township logo. founded in 1984 in a church longest serving employee artistic director",
            "exact_late_interaction_clause_text",
            match_all_filter(),
            SearchPlanSelectionRequest(
                1,
                2,
                best_effort_faithfulness_policy(),
                match_all_filter(),
            ),
        ),
    )

    assert_equal(
        response.selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(
        response.search.plan.stage2_operator.kind,
        "exact_late_interaction_clause_text",
    )
    assert_equal(response.search.hits[0].doc_id, "doc-answer")


def test_hosted_collection_runtime_executes_planned_search_with_native_goal() raises:
    var service_root = unique_service_root("kayak-service-runtime-planned-native")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
                    "beta",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish planned native fixture",
        ),
    )

    var response = execute_planned_search(
        ExactCpuBackend(),
        service_root,
        PlannedSearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            match_all_filter(),
            SearchPlanSelectionRequest(
                1,
                2,
                best_effort_faithfulness_policy(),
                match_all_filter(),
                "native_multivector",
            ),
        ),
    )

    assert_equal(
        response.selection.plan.candidate_generator.kind,
        "centroid_postings_imputed_flat",
    )
    assert_equal(response.search.hits[0].doc_id, "doc-a")


def test_hosted_collection_runtime_planned_debug_search_keeps_metadata_filter_guardrail() raises:
    var service_root = unique_service_root("kayak-service-runtime-planned-filter")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                    [DocumentMetadataUpdate("source", "wire")],
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
                    "beta",
                    [DocumentMetadataUpdate("source", "blog")],
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish planned filter fixture",
        ),
    )

    var response = execute_planned_debug_search(
        ExactCpuBackend(),
        service_root,
        PlannedSearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            one_of_filter("source", ["wire"]),
            SearchPlanSelectionRequest(
                1,
                2,
                best_effort_faithfulness_policy(),
                one_of_filter("source", ["wire"]),
                "native_multivector",
                True,
            ),
        ),
    )

    assert_equal(response.selection.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(response.debug.search.hits[0].doc_id, "doc-a")
    assert_equal(
        response.debug.explain.plan.candidate_generator.kind,
        "exact_full_scan",
    )


def test_hosted_collection_runtime_merges_metadata_and_filters_exactly() raises:
    var service_root = unique_service_root("kayak-service-runtime-metadata")

    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                    [
                        DocumentMetadataUpdate("source", "wire"),
                        DocumentMetadataUpdate("language", "en"),
                    ],
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
                    "beta",
                    [DocumentMetadataUpdate("source", "blog")],
                ),
            ],
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    [
                        DocumentMetadataUpdate("source", "analysis"),
                        DocumentMetadataUpdate("language", ""),
                    ],
                )
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish metadata fixture",
        ),
    )

    var metadata_response = execute_search(
        ExactCpuBackend(),
        service_root,
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            one_of_filter("source", ["analysis"]),
            exact_full_scan_search_plan(1, 1),
            False,
        ),
    )
    var removed_field_response = execute_search(
        ExactCpuBackend(),
        service_root,
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            one_of_filter("language", ["en"]),
            exact_full_scan_search_plan(1, 1),
            False,
        ),
    )

    assert_equal(len(metadata_response.hits), 1)
    assert_equal(metadata_response.hits[0].doc_id, "doc-a")
    assert_equal(len(removed_field_response.hits), 0)


def test_hosted_collection_runtime_supports_configured_gem_graph_stage1() raises:
    var service_root = unique_service_root("kayak-service-runtime-gem-policy")

    var collection_root = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
            SearchArtifactBuildPolicy(
                [gem_graph_build_spec(1, 1, 1, "gem_graph", 1, 1)]
            ),
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha",
                ),
                UpsertDocument(
                    make_document("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
                    "beta",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            "publish gem_graph fixture",
        ),
    )

    var response = execute_search(
        ExactCpuBackend(),
        service_root,
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            match_all_filter(),
            gem_graph_search_plan(
                1,
                2,
                best_effort_faithfulness_policy(),
                1,
                4,
            ),
            False,
        ),
    )
    var manifest = load_collection_manifest(collection_root)

    assert_equal(len(response.hits), 1)
    assert_equal(response.hits[0].doc_id, "doc-a")
    assert_equal(
        manifest.search_artifact_build_policy.stage1_artifacts[0].family,
        "gem_graph",
    )


def test_hosted_collection_lifecycle_report_uses_default_and_override_policy() raises:
    var service_root = unique_service_root("kayak-service-runtime-lifecycle")
    _ = build_three_snapshot_service_fixture(service_root, 1)

    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [UpsertDocument(make_document("doc-d", [[0.0, 1.0], [0.0, 1.0]]), "delta")],
        ),
    )

    var default_report = build_collection_lifecycle_report(
        service_root,
        CollectionLifecycleRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
        ),
    )
    var override_report = build_collection_lifecycle_report(
        service_root,
        CollectionLifecycleRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotRetentionPolicy(0, [SnapshotId("snapshot-0001")]),
        ),
    )

    assert_equal(default_report.latest_generation, 3)
    assert_equal(default_report.active_snapshot_id, "snapshot-0003")
    assert_equal(default_report.default_keep_latest_inactive_count, 1)
    assert_equal(default_report.effective_keep_latest_inactive_count, 1)
    assert_equal(default_report.reclaim_plan.reclaimable_snapshot_count, 1)
    assert_equal(default_report.draft_document_count, 4)
    assert_equal(default_report.pending_draft_mutation_count, 1)
    assert_equal(override_report.effective_keep_latest_inactive_count, 0)
    assert_equal(
        override_report.effective_pinned_snapshot_ids[0].value,
        "snapshot-0001",
    )
    assert_equal(override_report.reclaim_plan.reclaimable_snapshot_count, 1)
    assert_equal(
        override_report.reclaim_plan.decisions[1].reason,
        "inactive_reclaim_candidate",
    )
    assert_equal(
        override_report.reclaim_plan.decisions[2].reason,
        "pinned_snapshot",
    )


def test_hosted_reclaim_service_builds_and_executes_plan() raises:
    var service_root = unique_service_root("kayak-service-runtime-reclaim")
    var collection_root = build_three_snapshot_service_fixture(service_root, 1)

    var plan_response = build_reclaim_plan(
        service_root,
        BuildReclaimPlanRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
        ),
    )
    var dry_run = execute_reclaim(
        service_root,
        ExecuteReclaimRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            plan_response.plan,
        ),
    )
    assert_equal(plan_response.plan.reclaimable_snapshot_count, 1)
    assert_equal(plan_response.effective_keep_latest_inactive_count, 1)
    assert_equal(dry_run.result.applied, False)
    assert_equal((collection_root / "snapshots" / "snapshot-0001").exists(), True)
    assert_equal((collection_root / "segments" / "segment-1").exists(), True)

    var applied = execute_reclaim(
        service_root,
        ExecuteReclaimRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            plan_response.plan,
            False,
        ),
    )
    var after = build_reclaim_plan(
        service_root,
        BuildReclaimPlanRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
        ),
    )

    assert_equal(applied.result.applied, True)
    assert_equal(applied.result.snapshot_ids[0].value, "snapshot-0001")
    assert_equal((collection_root / "snapshots" / "snapshot-0001").exists(), False)
    assert_equal((collection_root / "segments" / "segment-1").exists(), False)
    assert_equal(after.plan.total_snapshot_count, 2)
    assert_equal(after.plan.reclaimable_snapshot_count, 0)


def test_retention_policy_update_persists_into_lifecycle_operations() raises:
    var service_root = unique_service_root("kayak-service-runtime-retention")
    var collection_root = build_three_snapshot_service_fixture(service_root, 1)

    var updated = update_collection_retention_policy(
        service_root,
        UpdateCollectionRetentionPolicyRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            2,
        ),
    )
    var manifest = load_collection_manifest(collection_root)
    var report = build_collection_lifecycle_report(
        service_root,
        CollectionLifecycleRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
        ),
    )

    assert_equal(updated.default_keep_latest_inactive_count, 2)
    assert_equal(manifest.default_keep_latest_inactive_count, 2)
    assert_equal(report.default_keep_latest_inactive_count, 2)
    assert_equal(report.effective_keep_latest_inactive_count, 2)
    assert_equal(report.reclaim_plan.reclaimable_snapshot_count, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
