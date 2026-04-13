from std.testing import TestSuite, assert_equal

from kayak import (
    BuildReclaimPlanRequest,
    BuildReclaimPlanResponse,
    CollectionHit,
    CollectionId,
    COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
    COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED,
    CollectionLifecycleRequest,
    CollectionLifecycleResponse,
    CollectionReclaimExecutionResult,
    CollectionReclaimPlan,
    CollectionSearchExplain,
    CreateCollectionRequest,
    DebugSearchResponse,
    DocumentMetadataUpdate,
    EncodedDocument,
    EncodedQuery,
    ExecuteReclaimRequest,
    ExecuteReclaimResponse,
    ExplainResponse,
    PlannedDebugSearchResponse,
    PlannedSearchRequest,
    PlannedSearchResponse,
    SearchPlanSelectionDecision,
    SearchPlanSelection,
    SearchPlanSelectionRequest,
    document_proxy_build_spec,
    gem_graph_build_spec,
    FaithfulnessAssessment,
    GraphSearchCounters,
    NamespaceId,
    ScoreHistogram,
    ScoreScalar,
    SearchResponse,
    SearchArtifactBuildPolicy,
    SearchStageProfile,
    SEARCH_PLAN_ORDER_POLICY_GOAL_DEFAULT,
    SEARCH_PLAN_SELECTION_CONSTRAINT_NONE,
    SEARCH_PLAN_SELECTION_OUTCOME_EXACT_FALLBACK_UNAVAILABLE,
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
    best_effort_faithfulness_policy,
    collection_lifecycle_request_json,
    collection_lifecycle_response_json,
    create_collection_request_json,
    debug_search_response_json,
    execute_reclaim_request_json,
    execute_reclaim_response_json,
    planned_debug_search_response_json,
    planned_search_request_json,
    planned_search_response_json,
    service_health_status_json,
    upsert_documents_request_json,
    update_collection_retention_policy_request_json,
    update_collection_retention_policy_response_json,
    layout_rooted_search_serving_scope,
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
        layout_rooted_search_serving_scope(),
        plan,
        CandidateSet(
            "exact_full_scan",
            hits.copy(),
            1,
            1,
            1,
            1,
            16,
            GraphSearchCounters(3, 5, 2, 1, 4),
        ),
        SearchStageProfile(
            "candidate_generation",
            1,
            1,
            1,
            1,
            1,
            1,
            16,
            GraphSearchCounters(3, 5, 2, 1, 4),
            ScoreHistogram(1, 1.0, 1.0, [1]),
        ),
        SearchStageProfile(
            "noop_topk",
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
            "none",
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            ScoreHistogram(1, 1.0, 1.0, [1]),
        ),
        SearchStageProfile(
            "exact_oracle",
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


def make_planned_search_request() raises -> PlannedSearchRequest:
    return PlannedSearchRequest(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
        "find the clause evidence",
        "",
        "clause_text",
        match_all_filter(),
        SearchPlanSelectionRequest(
            2,
            10,
            best_effort_faithfulness_policy(),
            match_all_filter(),
            "native_multivector",
            ["centroid_postings_imputed_flat", "document_proxy"],
            True,
            3,
            9,
        ),
    )


def make_planned_search_response() raises -> PlannedSearchResponse:
    var plan = exact_full_scan_search_plan(2, 2)
    var hits = [CollectionHit("segment-0001", "doc-a", ScoreScalar(1.0))]
    return PlannedSearchResponse(
        SearchPlanSelection(
            "balanced",
            "exact_fallback",
            ["exact_full_scan", "document_proxy"],
            ["document_proxy", "exact_full_scan"],
            layout_rooted_search_serving_scope(),
            plan,
            SearchPlanSelectionDecision(
                SEARCH_PLAN_ORDER_POLICY_GOAL_DEFAULT,
                SEARCH_PLAN_SELECTION_CONSTRAINT_NONE,
                SEARCH_PLAN_SELECTION_OUTCOME_EXACT_FALLBACK_UNAVAILABLE,
                "planner fell back to exact full scan for verification",
            ),
        ),
        SearchResponse(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            plan,
            hits.copy(),
        ),
    )


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
            SearchArtifactBuildPolicy(
                [document_proxy_build_spec("proxy_sidecar", 2)]
            ),
        )
    )

    assert_equal(json.find("\"collection_id\":\"news\"") != -1, True)
    assert_equal(
        json.find(
            "\"collection_layout_family\":\""
                + COLLECTION_LAYOUT_FAMILY_TENANT_ISOLATED
                + "\""
        ) != -1,
        True,
    )
    assert_equal(json.find("\"vector_dim\":128") != -1, True)
    assert_equal(
        json.find("\"default_keep_latest_inactive_count\":1") != -1,
        True,
    )
    assert_equal(
        json.find("\"search_artifact_build_policy\":[") != -1,
        True,
    )
    assert_equal(
        json.find("\"family\":\"document_proxy\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"key\":\"document_vector_budget\"") != -1,
        True,
    )
    assert_equal(
        json.find("\"value\":\"2\"") != -1,
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
            COLLECTION_LAYOUT_FAMILY_SHARED_POOL,
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
            3,
            "snapshot-0003",
            1,
            SearchArtifactBuildPolicy(
                [
                    document_proxy_build_spec("document_proxy", 2),
                    gem_graph_build_spec(2, 3, 1, "gem_graph", 4, 5),
                ]
            ),
            0,
            [SnapshotId("snapshot-0001")],
            4,
            1,
            sample_reclaim_plan(),
        )
    )
    assert_equal(
        lifecycle_response_json.find(
            "\"collection_layout_family\":\""
                + COLLECTION_LAYOUT_FAMILY_SHARED_POOL
                + "\""
        ) != -1,
        True,
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
        lifecycle_response_json.find("\"search_artifact_build_policy\":[") != -1,
        True,
    )
    assert_equal(
        lifecycle_response_json.find("\"key\":\"fine_cluster_count\"") != -1,
        True,
    )
    assert_equal(
        lifecycle_response_json.find("\"value\":\"5\"") != -1,
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
    assert_equal(
        json.find("\"stage2_reference_kind\":\"noop_topk\"") != -1,
        True,
    )
    assert_equal(json.find("\"stage2\":") != -1, True)
    assert_equal(json.find("\"stage_name\":\"noop_topk\"") != -1, True)
    assert_equal(json.find("\"tracks_graph_search\":true") != -1, True)
    assert_equal(json.find("\"tracks_graph_search\":false") != -1, True)
    assert_equal(json.find("\"materialized_artifacts\":[]") != -1, True)
    assert_equal(json.find("\"graph_search_counters\":") != -1, True)
    assert_equal(json.find("\"visited_vertex_count\":3") != -1, True)
    assert_equal(json.find("\"filter_application\":{") != -1, True)
    assert_equal(json.find("\"input_document_count\":1") != -1, True)
    assert_equal(json.find("\"selectivity\":1.0") != -1, True)


def test_planned_search_json_surfaces_selection_and_planning_contract() raises:
    var request_json = planned_search_request_json(make_planned_search_request())
    var response_json = planned_search_response_json(make_planned_search_response())
    var debug_json = planned_debug_search_response_json(
        PlannedDebugSearchResponse(
            make_planned_search_response().selection,
            make_debug_response(),
        )
    )

    assert_equal(request_json.find("\"planning\":") != -1, True)
    assert_equal(request_json.find("\"goal\":\"native_multivector\"") != -1, True)
    assert_equal(
        request_json.find("\"query_text\":\"find the clause evidence\"") != -1,
        True,
    )
    assert_equal(
        request_json.find("\"stage2_reference_kind\":\"\"") != -1,
        True,
    )
    assert_equal(
        request_json.find("\"stage3_verifier_kind\":\"clause_text\"") != -1,
        True,
    )
    assert_equal(
        request_json.find("\"preferred_candidate_generator_kinds\":[\"centroid_postings_imputed_flat\",\"document_proxy\"]")
            != -1,
        True,
    )
    assert_equal(request_json.find("\"graph_cluster_top_k_per_query_token\":3") != -1, True)
    assert_equal(request_json.find("\"graph_beam_width\":9") != -1, True)
    assert_equal(response_json.find("\"selection\":") != -1, True)
    assert_equal(
        response_json.find("\"available_candidate_generator_kinds\":[\"exact_full_scan\",\"document_proxy\"]")
            != -1,
        True,
    )
    assert_equal(
        response_json.find("\"effective_candidate_generator_order\":[\"document_proxy\",\"exact_full_scan\"]")
            != -1,
        True,
    )
    assert_equal(
        response_json.find("\"serving_scope_kind\":\"layout_rooted\"") != -1,
        True,
    )
    assert_equal(
        response_json.find("\"serving_scope_requires_logical_pushdown\":false")
            != -1,
        True,
    )
    assert_equal(response_json.find("\"decision\":") != -1, True)
    assert_equal(
        response_json.find("\"order_policy_kind\":\"goal_default\"") != -1,
        True,
    )
    assert_equal(
        response_json.find("\"outcome_kind\":\"exact_fallback_unavailable\"") != -1,
        True,
    )
    assert_equal(debug_json.find("\"selection\":") != -1, True)
    assert_equal(debug_json.find("\"debug\":") != -1, True)
    assert_equal(
        debug_json.find("\"serving_scope_kind\":\"layout_rooted\"") != -1,
        True,
    )


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
