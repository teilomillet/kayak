from std.testing import TestSuite, assert_equal

from kayak import (
    BuildReclaimPlanRequest,
    BuildReclaimPlanResponse,
    CollectionHit,
    CollectionId,
    CollectionLifecycleRequest,
    CollectionLifecycleResponse,
    CollectionReclaimExecutionResult,
    CollectionReclaimPlan,
    CollectionSearchExplain,
    CreateCollectionRequest,
    DebugSearchResponse,
    DocumentMetadataUpdate,
    EncodedDocument,
    ExecuteReclaimRequest,
    ExecuteReclaimResponse,
    ExplainResponse,
    FaithfulnessAssessment,
    NamespaceId,
    ScoreHistogram,
    ScoreScalar,
    SearchResponse,
    SearchStageProfile,
    SegmentId,
    service_metrics_snapshot_json,
    SnapshotId,
    SnapshotRetentionDecision,
    SnapshotRetentionPolicy,
    TenantId,
    UpdateCollectionRetentionPolicyRequest,
    UpdateCollectionRetentionPolicyResponse,
    VECTOR_SCALAR_NAME,
    UpsertDocument,
    UpsertDocumentsRequest,
    build_reclaim_plan_request_json,
    build_reclaim_plan_response_json,
    collection_lifecycle_request_json,
    collection_lifecycle_response_json,
    create_collection_request_json,
    debug_search_response_json,
    execute_reclaim_request_json,
    execute_reclaim_response_json,
    service_health_status_json,
    upsert_documents_request_json,
    update_collection_retention_policy_request_json,
    update_collection_retention_policy_response_json,
)
from kayak.filters import match_all_filter
from kayak.planning import CandidateSet, exact_full_scan_search_plan
from kayak.service import ServiceHealthStatus, ServiceMetricsSnapshot


def make_debug_response() raises -> DebugSearchResponse:
    var plan = exact_full_scan_search_plan(2, 2)
    var hits = [CollectionHit("segment-0001", "doc-a", ScoreScalar(1.0))]
    var search = SearchResponse(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        plan,
        hits.copy(),
    )
    var explain = CollectionSearchExplain(
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
    return DebugSearchResponse(search, explain)


def sample_reclaim_plan() raises -> CollectionReclaimPlan:
    return CollectionReclaimPlan(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        "snapshot-0003",
        3,
        2,
        1,
        1,
        1,
        1024,
        [
            SnapshotRetentionDecision(
                SnapshotId("snapshot-0003"),
                3,
                1,
                1024,
                True,
                "active_snapshot",
            ),
            SnapshotRetentionDecision(
                SnapshotId("snapshot-0002"),
                2,
                1,
                1024,
                True,
                "retained_inactive_by_generation",
            ),
            SnapshotRetentionDecision(
                SnapshotId("snapshot-0001"),
                1,
                1,
                1024,
                False,
                "inactive_reclaim_candidate",
            ),
        ],
        [SegmentId("segment-1")],
    )


def sample_reclaim_execution_result() raises -> CollectionReclaimExecutionResult:
    return CollectionReclaimExecutionResult(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        True,
        1,
        1,
        1024,
        [SnapshotId("snapshot-0001")],
        [SegmentId("segment-1")],
    )


def test_create_collection_request_json_is_machine_readable() raises:
    var json = create_collection_request_json(
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
        )
    )

    assert_equal(json.find("\"collection_id\":\"news\"") != -1, True)
    assert_equal(json.find("\"vector_dim\":128") != -1, True)
    assert_equal(
        json.find("\"default_keep_latest_inactive_count\":1") != -1,
        True,
    )


def test_lifecycle_and_reclaim_json_are_machine_readable() raises:
    var lifecycle_request_json = collection_lifecycle_request_json(
        CollectionLifecycleRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotRetentionPolicy(0, [SnapshotId("snapshot-0001")]),
        )
    )
    var lifecycle_response_json = collection_lifecycle_response_json(
        CollectionLifecycleResponse(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            3,
            "snapshot-0003",
            1,
            0,
            [SnapshotId("snapshot-0001")],
            4,
            1,
            sample_reclaim_plan(),
        )
    )
    var build_request_json = build_reclaim_plan_request_json(
        BuildReclaimPlanRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotRetentionPolicy(0, [SnapshotId("snapshot-0001")]),
        )
    )
    var build_response_json = build_reclaim_plan_response_json(
        BuildReclaimPlanResponse(
            sample_reclaim_plan(),
            0,
            [SnapshotId("snapshot-0001")],
        )
    )
    var execute_request_json = execute_reclaim_request_json(
        ExecuteReclaimRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            sample_reclaim_plan(),
            False,
        )
    )
    var execute_response_json = execute_reclaim_response_json(
        ExecuteReclaimResponse(sample_reclaim_execution_result())
    )

    assert_equal(lifecycle_request_json.find("\"has_policy_override\":true") != -1, True)
    assert_equal(
        lifecycle_request_json.find("\"policy_override_pinned_snapshot_ids\":[\"snapshot-0001\"]")
            != -1,
        True,
    )
    assert_equal(
        lifecycle_response_json.find("\"default_keep_latest_inactive_count\":1") != -1,
        True,
    )
    assert_equal(
        lifecycle_response_json.find("\"effective_keep_latest_inactive_count\":0") != -1,
        True,
    )
    assert_equal(
        lifecycle_response_json.find("\"reclaim_plan\":") != -1,
        True,
    )
    assert_equal(
        build_request_json.find("\"policy_override_keep_latest_inactive_count\":0")
            != -1,
        True,
    )
    assert_equal(
        build_response_json.find("\"effective_pinned_snapshot_ids\":[\"snapshot-0001\"]")
            != -1,
        True,
    )
    assert_equal(execute_request_json.find("\"dry_run\":false") != -1, True)
    assert_equal(
        execute_response_json.find("\"result\":") != -1,
        True,
    )
    assert_equal(
        execute_response_json.find("\"applied\":true") != -1,
        True,
    )


def test_retention_policy_update_json_is_machine_readable() raises:
    var request_json = update_collection_retention_policy_request_json(
        UpdateCollectionRetentionPolicyRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            2,
        )
    )
    var response_json = update_collection_retention_policy_response_json(
        UpdateCollectionRetentionPolicyResponse(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            3,
            "snapshot-0003",
            2,
        )
    )

    assert_equal(
        request_json.find("\"default_keep_latest_inactive_count\":2") != -1,
        True,
    )
    assert_equal(response_json.find("\"latest_generation\":3") != -1, True)
    assert_equal(
        response_json.find("\"active_snapshot_id\":\"snapshot-0003\"") != -1,
        True,
    )


def test_debug_search_response_json_embeds_explain_payload() raises:
    var json = debug_search_response_json(make_debug_response())

    assert_equal(json.find("\"search\":") != -1, True)
    assert_equal(json.find("\"debug\":") != -1, True)
    assert_equal(json.find("\"candidate_generator_kind\":\"exact_full_scan\"") != -1, True)
    assert_equal(json.find("\"faithfulness_policy_kind\":\"exact_stage1_required\"") != -1, True)
    assert_equal(json.find("\"faithfulness\":") != -1, True)
    assert_equal(json.find("\"candidate_stage\":") != -1, True)


def test_service_health_status_json_contains_counters() raises:
    var json = service_health_status_json(ServiceHealthStatus("ok", 2, 3, 1))

    assert_equal(json.find("\"status\":\"ok\"") != -1, True)
    assert_equal(json.find("\"live_snapshot_count\":3") != -1, True)


def test_service_metrics_snapshot_json_contains_operational_counters() raises:
    var json = service_metrics_snapshot_json(
        ServiceMetricsSnapshot(2, 3, 10, 80, 4096, 4, 2, 3, 1024, 1, 5)
    )

    assert_equal(json.find("\"published_snapshot_count\":4") != -1, True)
    assert_equal(json.find("\"inactive_snapshot_count\":2") != -1, True)
    assert_equal(json.find("\"inactive_unique_segment_count\":3") != -1, True)
    assert_equal(json.find("\"inactive_unique_byte_size\":1024") != -1, True)
    assert_equal(json.find("\"pending_draft_collection_count\":1") != -1, True)
    assert_equal(json.find("\"pending_draft_mutation_count\":5") != -1, True)


def test_upsert_documents_request_json_contains_metadata_updates() raises:
    var json = upsert_documents_request_json(
        UpsertDocumentsRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            [
                UpsertDocument(
                    EncodedDocument("doc-a", [[1.0, 0.0]]),
                    "alpha",
                    [DocumentMetadataUpdate("source", "wire")],
                )
            ],
        )
    )

    assert_equal(json.find("\"has_metadata_updates\":true") != -1, True)
    assert_equal(json.find("\"metadata_updates\"") != -1, True)
    assert_equal(json.find("\"key\":\"source\"") != -1, True)
    assert_equal(json.find("\"value\":\"wire\"") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
