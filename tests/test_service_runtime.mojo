from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    DeleteDocumentsRequest,
    DocumentMetadataUpdate,
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
    NamespaceId,
    SnapshotId,
    TenantId,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    best_effort_faithfulness_policy,
    build_service_health_status,
    build_service_metrics_snapshot,
    create_collection,
    create_snapshot,
    default_exact_search_request,
    delete_documents,
    document_proxy_search_plan,
    execute_debug_search,
    execute_search,
    exact_full_scan_search_plan,
    export_snapshot,
    import_snapshot,
    load_collection_manifest,
    one_of_filter,
    SearchRequest,
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

    var raised = False
    try:
        _ = execute_search(
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
    except:
        raised = True

    assert_equal(exact_filtered.hits[0].doc_id, "doc-a")
    assert_equal(raised, True)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
