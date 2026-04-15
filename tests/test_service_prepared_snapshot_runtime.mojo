from std.collections import List
from std.testing import TestSuite, assert_equal
from std.pathlib import Path

from kayak import (
    CollectionId,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    NamespaceId,
    PlannedSearchRequest,
    SearchPlanSelectionRequest,
    SnapshotId,
    TenantId,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    best_effort_faithfulness_policy,
    create_collection,
    create_snapshot,
    default_exact_search_request,
    execute_planned_search,
    execute_planned_search_with_prepared_snapshot,
    execute_search,
    execute_search_with_prepared_snapshot,
    match_all_filter,
    prepare_service_search_snapshot,
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
    doc_id: String, vectors: List[List[Float32]]
) raises -> EncodedDocument:
    return EncodedDocument(doc_id, vectors.copy())


def build_service_fixture(service_root: Path) raises:
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
                    make_document("doc-old", [[1.0, 0.0], [0.0, 1.0]]),
                    "old snapshot document",
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
            "publish first snapshot",
        ),
    )


def test_prepared_snapshot_stays_pinned_after_new_snapshot_publish() raises:
    var service_root = unique_service_root("kayak-service-prepared-snapshot-pinned")
    build_service_fixture(service_root)
    var backend = ExactCpuBackend()
    var prepared_snapshot = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        False,
    )
    var request = default_exact_search_request(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        make_query(),
        1,
        "colbertv2",
    )

    var baseline = execute_search_with_prepared_snapshot(
        backend,
        prepared_snapshot,
        request,
    )
    assert_equal(baseline.hits[0].doc_id, "doc-old")

    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    make_document("doc-new", [[2.0, 0.0], [0.0, 2.0]]),
                    "new snapshot document",
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
            SnapshotId("snapshot-0002"),
            "publish second snapshot",
        ),
    )

    var still_pinned = execute_search_with_prepared_snapshot(
        backend,
        prepared_snapshot,
        request,
    )
    assert_equal(still_pinned.snapshot_id.value, "snapshot-0001")
    assert_equal(still_pinned.hits[0].doc_id, "doc-old")

    var updated_request = default_exact_search_request(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0002"),
        make_query(),
        1,
        "colbertv2",
    )
    var updated = execute_search(
        backend,
        service_root,
        updated_request,
    )
    assert_equal(updated.hits[0].doc_id, "doc-new")

    var prepared_updated = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0002"),
        False,
    )
    var updated_from_prepared = execute_search_with_prepared_snapshot(
        backend,
        prepared_updated,
        updated_request,
    )
    assert_equal(updated_from_prepared.hits[0].doc_id, "doc-new")


def test_planned_search_with_prepared_snapshot_matches_stateless_runtime() raises:
    var service_root = unique_service_root("kayak-service-prepared-snapshot-planned")
    build_service_fixture(service_root)
    var backend = ExactCpuBackend()
    var request = PlannedSearchRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        make_query(),
        "colbertv2",
        match_all_filter(),
        SearchPlanSelectionRequest(
            1,
            4,
            best_effort_faithfulness_policy(),
            match_all_filter(),
        ),
    )

    var stateless = execute_planned_search(
        backend,
        service_root,
        request,
    )
    var prepared_snapshot = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        True,
    )
    var prepared = execute_planned_search_with_prepared_snapshot(
        backend,
        prepared_snapshot,
        request,
    )

    assert_equal(
        prepared.selection.plan.candidate_generator.kind,
        stateless.selection.plan.candidate_generator.kind,
    )
    assert_equal(
        prepared.search.plan.candidate_generator.kind,
        stateless.search.plan.candidate_generator.kind,
    )
    assert_equal(len(prepared.search.hits), len(stateless.search.hits))
    assert_equal(prepared.search.hits[0].doc_id, stateless.search.hits[0].doc_id)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
