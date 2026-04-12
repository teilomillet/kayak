from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    DeleteDocumentsRequest,
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
    create_collection,
    create_snapshot,
    default_exact_search_request,
    delete_documents,
    execute_debug_search,
    execute_search,
    export_snapshot,
    import_snapshot,
    load_collection_manifest,
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
    assert_equal(search_response.hits[0].doc_id, "doc-a")
    assert_equal(debug_response.explain.final_hits[0].doc_id, "doc-a")
    assert_equal(debug_response.explain.snapshot_id, "snapshot-0001")
    assert_equal(exported.snapshot_id.value, "snapshot-0001")
    assert_equal(imported.snapshot_id.value, "snapshot-0001")
    assert_equal(imported_search.hits[0].doc_id, "doc-a")


def test_hosted_collection_runtime_persists_append_only_draft_mutations() raises:
    var service_root = unique_service_root("kayak-service-runtime-draft-log")

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
    assert_equal(draft_manifest.find("mutation_count\t3") >= 0, True)
    assert_equal(
        (collection_root / "draft" / "mutations" / "mutation-1" / "manifest.tsv").exists(),
        True,
    )
    assert_equal(
        (collection_root / "draft" / "mutations" / "mutation-2" / "manifest.tsv").exists(),
        True,
    )
    assert_equal(
        (collection_root / "draft" / "mutations" / "mutation-3" / "manifest.tsv").exists(),
        True,
    )
    assert_equal(
        (collection_root / "draft" / "packed_index" / "manifest.tsv").exists(),
        False,
    )
    assert_equal(document_count_after_delete, 1)
    assert_equal(snapshot.stats.document_count, 1)
    assert_equal(snapshot.segment_ids[0].value, "segment-1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
