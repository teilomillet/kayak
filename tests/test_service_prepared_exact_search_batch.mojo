from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionHit,
    CollectionId,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    EncodedDocument,
    EncodedQuery,
    ExactScoringConfig,
    NamespaceId,
    PreparedExactSearchBatchConfig,
    SearchRequest,
    SearchResponse,
    SnapshotId,
    TenantId,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    default_exact_search_request,
    exact_cpu_backend_for_scoring_config,
    execute_search_batch_with_prepared_snapshot,
    execute_search_with_prepared_snapshot,
    prepare_service_search_snapshot,
    create_collection,
    create_snapshot,
    upsert_documents,
)


def unique_service_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def make_basis_vector(
    first: Float32,
    second: Float32,
    third: Float32,
    fourth: Float32,
) -> List[Float32]:
    var values = List[Float32]()
    values.reserve(128)
    values.append(first)
    values.append(second)
    values.append(third)
    values.append(fourth)
    for _ in range(4, 128):
        values.append(0.0)
    return values^


def make_query(
    first_a: Float32,
    second_a: Float32,
    third_a: Float32,
    fourth_a: Float32,
    first_b: Float32,
    second_b: Float32,
    third_b: Float32,
    fourth_b: Float32,
) raises -> EncodedQuery:
    return EncodedQuery(
        [
            make_basis_vector(first_a, second_a, third_a, fourth_a),
            make_basis_vector(first_b, second_b, third_b, fourth_b),
        ]
    )


def make_document(
    doc_id: String,
    first: List[Float32],
    second: List[Float32],
) raises -> EncodedDocument:
    return EncodedDocument(doc_id, [first.copy(), second.copy()])


def build_service_fixture(service_root: Path) raises:
    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
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
                    make_document(
                        "doc-a",
                        make_basis_vector(1.0, 0.0, 0.0, 0.0),
                        make_basis_vector(0.0, 1.0, 0.0, 0.0),
                    ),
                    "alpha",
                ),
                UpsertDocument(
                    make_document(
                        "doc-b",
                        make_basis_vector(0.0, 1.0, 0.0, 0.0),
                        make_basis_vector(0.0, 0.0, 1.0, 0.0),
                    ),
                    "beta",
                ),
                UpsertDocument(
                    make_document(
                        "doc-c",
                        make_basis_vector(0.0, 0.0, 1.0, 0.0),
                        make_basis_vector(0.0, 0.0, 0.0, 1.0),
                    ),
                    "gamma",
                ),
                UpsertDocument(
                    make_document(
                        "doc-d",
                        make_basis_vector(1.0, 0.0, 0.0, 0.0),
                        make_basis_vector(0.0, 0.0, 0.0, 1.0),
                    ),
                    "delta",
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


def build_requests() raises -> List[SearchRequest]:
    var base = List[SearchRequest]()
    base.append(
        default_exact_search_request(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            4,
            "colbertv2",
        )
    )
    base.append(
        default_exact_search_request(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            4,
            "colbertv2",
        )
    )
    base.append(
        default_exact_search_request(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            4,
            "colbertv2",
        )
    )
    base.append(
        default_exact_search_request(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            4,
            "colbertv2",
        )
    )

    var requests = List[SearchRequest]()
    for _ in range(2):
        for request in base:
            requests.append(request.copy())

    return requests^


def execute_serial_batch(
    read requests: List[SearchRequest],
    read scoring_config: ExactScoringConfig,
    service_root: Path,
) raises -> List[SearchResponse]:
    var prepared = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        False,
    )
    var backend = exact_cpu_backend_for_scoring_config(scoring_config)
    var responses = List[SearchResponse]()
    for request in requests:
        responses.append(
            execute_search_with_prepared_snapshot(backend, prepared, request)
        )

    return responses^


def assert_search_responses_equal(
    read expected: List[SearchResponse], read observed: List[SearchResponse]
) raises:
    assert_equal(len(expected), len(observed))
    for response_index in range(len(expected)):
        assert_equal(
            expected[response_index].snapshot_id.value,
            observed[response_index].snapshot_id.value,
        )
        assert_equal(
            len(expected[response_index].hits),
            len(observed[response_index].hits),
        )
        for hit_index in range(len(expected[response_index].hits)):
            var expected_hit = expected[response_index].hits[hit_index].copy()
            var observed_hit = observed[response_index].hits[hit_index].copy()
            assert_equal(expected_hit.doc_id, observed_hit.doc_id)
            assert_equal(expected_hit.score, observed_hit.score)


def test_search_batch_with_prepared_snapshot_matches_serial_default() raises:
    var service_root = unique_service_root("kayak-service-prepared-search-batch-default")
    build_service_fixture(service_root)
    var requests = build_requests()
    var prepared = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        False,
    )

    var expected = execute_serial_batch(
        requests,
        ExactScoringConfig(),
        service_root,
    )
    var observed = execute_search_batch_with_prepared_snapshot(
        prepared,
        requests,
        4,
    )

    assert_search_responses_equal(expected, observed)


def test_search_batch_with_prepared_snapshot_matches_serial_override() raises:
    var service_root = unique_service_root("kayak-service-prepared-search-batch-override")
    build_service_fixture(service_root)
    var requests = build_requests()
    var scoring_config = ExactScoringConfig()
    scoring_config.parallel_work_item_count_override = 2
    var prepared = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        False,
    )

    var expected = execute_serial_batch(requests, scoring_config, service_root)
    var observed = execute_search_batch_with_prepared_snapshot(
        prepared,
        requests,
        PreparedExactSearchBatchConfig(4, scoring_config),
    )

    assert_search_responses_equal(expected, observed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
