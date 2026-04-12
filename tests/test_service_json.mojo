from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionHit,
    CollectionId,
    CollectionSearchExplain,
    CreateCollectionRequest,
    DebugSearchResponse,
    ExplainResponse,
    NamespaceId,
    ScoreHistogram,
    ScoreScalar,
    SearchResponse,
    SearchStageProfile,
    SnapshotId,
    TenantId,
    VECTOR_SCALAR_NAME,
    create_collection_request_json,
    debug_search_response_json,
    service_health_status_json,
)
from kayak.filters import match_all_filter
from kayak.planning import CandidateSet, exact_full_scan_search_plan
from kayak.service import ServiceHealthStatus


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
        hits^,
    )
    return DebugSearchResponse(search, explain)


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


def test_debug_search_response_json_embeds_explain_payload() raises:
    var json = debug_search_response_json(make_debug_response())

    assert_equal(json.find("\"search\":") != -1, True)
    assert_equal(json.find("\"debug\":") != -1, True)
    assert_equal(json.find("\"candidate_generator_kind\":\"exact_full_scan\"") != -1, True)
    assert_equal(json.find("\"candidate_stage\":") != -1, True)


def test_service_health_status_json_contains_counters() raises:
    var json = service_health_status_json(ServiceHealthStatus("ok", 2, 3, 1))

    assert_equal(json.find("\"status\":\"ok\"") != -1, True)
    assert_equal(json.find("\"live_snapshot_count\":3") != -1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
