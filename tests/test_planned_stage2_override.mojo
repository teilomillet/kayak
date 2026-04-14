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
    execute_planned_debug_search,
    one_of_filter,
    upsert_documents,
)


def unique_service_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def make_document(
    doc_id: String, vectors: List[List[Float32]]
) raises -> EncodedDocument:
    return EncodedDocument(doc_id, vectors.copy())


def test_planned_clause_text_override_preserves_exact_candidate_window() raises:
    var service_root = unique_service_root(
        "kayak-service-runtime-planned-clause-text-focused"
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
            "publish planned clause-text focused fixture",
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
            EncodedQuery([[1.0, 0.0], [1.0, 0.0]]),
            "colbertv2",
            "Gugulethu township logo. founded in 1984 in a church longest serving employee artistic director",
            "",
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

    assert_equal(response.selection.plan.candidate_budget.candidate_k, 2)
    assert_equal(response.debug.search.plan.candidate_budget.candidate_k, 2)
    assert_equal(len(response.debug.explain.candidate_set.hits), 2)
    assert_equal(response.debug.explain.candidate_set.hits[0].doc_id, "doc-context")
    assert_equal(response.debug.search.hits[0].doc_id, "doc-answer")
    assert_equal(response.debug.explain.final_hits[0].doc_id, "doc-answer")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
