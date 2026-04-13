from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionHit,
    CollectionId,
    CollectionSearchExplain,
    CreateCollectionRequest,
    DebugSearchResponse,
    DeleteDocumentsRequest,
    EncodedDocument,
    EncodedQuery,
    ExplainRequest,
    ExplainResponse,
    DocumentMetadataUpdate,
    NamespaceId,
    ScoreScalar,
    SearchRequest,
    SearchResponse,
    ScoreHistogram,
    SearchStageProfile,
    SegmentId,
    SnapshotId,
    TenantId,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    FaithfulnessAssessment,
    best_effort_faithfulness_policy,
    default_exact_search_request,
    document_proxy_search_plan,
    oracle_full_recall_required_faithfulness_policy,
)
from kayak.filters import match_all_filter
from kayak.planning import CandidateSet, exact_full_scan_search_plan
from kayak.service import (
    CreateSnapshotRequest,
    ExportSnapshotRequest,
    ImportSnapshotRequest,
    ServiceHealthStatus,
    ServiceMetricsSnapshot,
)


def make_vector(value: Float64) -> List[Float32]:
    return [Float32(value), Float32(value + 1.0)]


def make_query() raises -> EncodedQuery:
    return EncodedQuery([make_vector(1.0)])


def make_document(doc_id: String, value: Float64) raises -> EncodedDocument:
    return EncodedDocument(doc_id, [make_vector(value)])


def make_explain() raises -> CollectionSearchExplain:
    var plan = exact_full_scan_search_plan(2, 2)
    var hits = [CollectionHit("segment-0001", "doc-a", ScoreScalar(1.0))]

    return CollectionSearchExplain(
        "news",
        "snapshot-0001",
        plan,
        CandidateSet("exact_full_scan", hits.copy(), 1, 1, 1, 1, 16),
        SearchStageProfile(
            "candidate_generation",
            1,
            1,
            1,
            1,
            1,
            1,
            16,
            ScoreHistogram(1, 1.0, 1.0, [1]),
        ),
        SearchStageProfile(
            "exact_late_interaction",
            1,
            1,
            1,
            1,
            1,
            1,
            16,
            ScoreHistogram(1, 1.0, 1.0, [1]),
        ),
        1.0,
        FaithfulnessAssessment(
            "exact_stage1_required",
            "exact_stage1",
            True,
            True,
            1.0,
            True,
            "faithfulness policy satisfied: stage 1 is exact",
        ),
        hits^,
    )


def test_create_collection_request_materializes_manifest() raises:
    var request = CreateCollectionRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "colbertv2",
        VECTOR_SCALAR_NAME,
        128,
    )
    var manifest = request.to_manifest()

    assert_equal(manifest.collection_id.value, "news")
    assert_equal(manifest.tenant_id.value, "tenant-a")
    assert_equal(manifest.latest_generation, 0)


def test_document_mutation_requests_keep_text_sidecar_explicit() raises:
    var request = UpsertDocumentsRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        [
            UpsertDocument(
                make_document("doc-a", 1.0),
                "alpha",
                [DocumentMetadataUpdate("source", "wire")],
            ),
            UpsertDocument(make_document("doc-b", 2.0)),
        ],
    )
    var delete_request = DeleteDocumentsRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        ["doc-a"],
    )

    assert_equal(len(request.documents), 2)
    assert_equal(request.documents[0].has_text, True)
    assert_equal(request.documents[0].has_metadata_updates, True)
    assert_equal(request.documents[0].metadata_updates[0].key, "source")
    assert_equal(request.documents[1].has_text, False)
    assert_equal(request.documents[1].has_metadata_updates, False)
    assert_equal(delete_request.doc_ids[0], "doc-a")


def test_default_search_request_builds_exact_plan() raises:
    var request = default_exact_search_request(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        make_query(),
        3,
        True,
    )

    assert_equal(request.snapshot_id.value, "snapshot-0001")
    assert_equal(request.filter_expression.is_match_all(), True)
    assert_equal(request.plan.candidate_generator.kind, "exact_full_scan")
    assert_equal(request.plan.faithfulness_policy.kind, "exact_stage1_required")
    assert_equal(request.plan.candidate_budget.final_k, 3)
    assert_equal(request.debug_mode, True)


def test_search_request_rejects_unverifiable_oracle_guardrail_without_debug() raises:
    var raised = False

    try:
        _ = SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(),
            match_all_filter(),
            document_proxy_search_plan(
                2,
                2,
                oracle_full_recall_required_faithfulness_policy(),
            ),
            False,
        )
    except:
        raised = True

    assert_equal(raised, True)


def test_search_request_allows_best_effort_approximate_search_without_debug() raises:
    var request = SearchRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        make_query(),
        match_all_filter(),
        document_proxy_search_plan(2, 2, best_effort_faithfulness_policy()),
        False,
    )

    assert_equal(request.plan.faithfulness_policy.kind, "best_effort")


def test_search_and_debug_responses_match_explain_scope() raises:
    var request = SearchRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        make_query(),
        match_all_filter(),
        exact_full_scan_search_plan(2, 2),
        True,
    )
    var hits = [CollectionHit("segment-0001", "doc-a", ScoreScalar(1.0))]
    var response = SearchResponse(
        request.collection_id,
        request.tenant_id,
        request.namespace_id,
        request.snapshot_id,
        request.plan,
        hits.copy(),
    )
    var explain = make_explain()
    var debug_response = DebugSearchResponse(response, explain)
    var explain_request = ExplainRequest(request)
    var explain_response = ExplainResponse(explain)

    assert_equal(response.hits[0].doc_id, "doc-a")
    assert_equal(debug_response.explain.snapshot_id, "snapshot-0001")
    assert_equal(explain_request.search.collection_id.value, "news")
    assert_equal(explain_response.explain.collection_id, "news")


def test_snapshot_and_status_contracts_hold_service_metadata() raises:
    var snapshot_request = CreateSnapshotRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0002"),
        "seal newly ingested documents",
    )
    var export_request = ExportSnapshotRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0002"),
    )
    var import_request = ImportSnapshotRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0003"),
        "file:///tmp/news-snapshot.tar",
    )
    var health = ServiceHealthStatus("ok", 1, 1, 0)
    var metrics = ServiceMetricsSnapshot(1, 2, 10, 80, 4096)

    assert_equal(snapshot_request.reason, "seal newly ingested documents")
    assert_equal(export_request.snapshot_id.value, "snapshot-0002")
    assert_equal(import_request.source_uri, "file:///tmp/news-snapshot.tar")
    assert_equal(health.status, "ok")
    assert_equal(metrics.vector_count, 80)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
