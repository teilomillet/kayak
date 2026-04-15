from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    EncodedDocument,
    EncodedQuery,
    ExactScoringConfig,
    NamespaceId,
    PreparedExactSearchBatchConfig,
    SearchPlan,
    SearchRequest,
    SearchResponse,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    VECTOR_SCALAR_NAME,
    best_effort_faithfulness_policy,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    ensure_one_segment_collection_mirror,
    exact_cpu_backend_for_scoring_config,
    execute_search_batch_with_prepared_snapshot,
    execute_search_with_prepared_snapshot,
    match_all_filter,
    pack_documents,
    prepare_service_search_snapshot,
)
from kayak.service.paths import service_collection_root


def unique_service_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def make_basis_vector(dim_index: Int, scale: Float32 = 1.0) -> List[Float32]:
    var values = List[Float32]()
    values.reserve(128)

    for dim in range(128):
        if dim == dim_index:
            values.append(scale)
        else:
            values.append(0.0)

    return values^


def make_document_block(
    doc_id: String, start_dim: Int, scale: Float32
) raises -> EncodedDocument:
    var token_vectors = List[List[Float32]]()

    for token_index in range(32):
        token_vectors.append(make_basis_vector(start_dim + token_index, scale))

    return EncodedDocument(doc_id, token_vectors^)


def make_query_block(start_dim: Int, scale: Float32 = 1.0) raises -> EncodedQuery:
    var token_vectors = List[List[Float32]]()

    for token_index in range(32):
        token_vectors.append(make_basis_vector(start_dim + token_index, scale))

    return EncodedQuery(token_vectors^)


def build_service_fixture(service_root: Path) raises:
    var stored_index = StoredPackedIndex(
        "mock://prepared-batch-segment-mirror",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        pack_documents(
            [
                make_document_block("doc-a", 0, 1.0),
                make_document_block("doc-b", 0, 0.9),
                make_document_block("doc-c", 32, 1.0),
                make_document_block("doc-d", 0, 0.5),
            ]
        ),
    )
    var collection_root = service_collection_root(
        service_root,
        TenantId("tenant-a"),
        NamespaceId("search"),
        CollectionId("news"),
    )
    _ = ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        1,
        stored_index,
        0,
        0,
        16,
    )


def candidate_k() -> Int:
    return 4


def build_requests(read plan: SearchPlan) raises -> List[SearchRequest]:
    var base = List[SearchRequest]()
    base.append(
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query_block(0),
            "colbertv2",
            match_all_filter(),
            plan,
            False,
        )
    )
    base.append(
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query_block(0, 0.9),
            "colbertv2",
            match_all_filter(),
            plan,
            False,
        )
    )
    base.append(
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query_block(32),
            "colbertv2",
            match_all_filter(),
            plan,
            False,
        )
    )
    base.append(
        SearchRequest(
            CollectionId("news"),
            TenantId("tenant-a"),
            NamespaceId("search"),
            SnapshotId("snapshot-0001"),
            make_query_block(0, 0.5),
            "colbertv2",
            match_all_filter(),
            plan,
            False,
        )
    )

    var requests = List[SearchRequest]()
    for _ in range(2):
        for request in base:
            requests.append(request.copy())

    return requests^


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
            assert_equal(
                expected[response_index].hits[hit_index].doc_id,
                observed[response_index].hits[hit_index].doc_id,
            )
            assert_equal(
                expected[response_index].hits[hit_index].score,
                observed[response_index].hits[hit_index].score,
            )


def assert_segment_mirror_batch_matches_serial(
    service_root: Path,
    read plan: SearchPlan,
    read scoring_config: ExactScoringConfig,
) raises:
    var requests = build_requests(plan)
    var baseline = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        False,
        False,
    )
    var mirrored = prepare_service_search_snapshot(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        False,
        True,
    )
    var backend = exact_cpu_backend_for_scoring_config(scoring_config)
    var expected = List[SearchResponse]()
    var mirrored_serial = List[SearchResponse]()

    for request in requests:
        expected.append(
            execute_search_with_prepared_snapshot(backend, baseline, request)
        )
        mirrored_serial.append(
            execute_search_with_prepared_snapshot(backend, mirrored, request)
        )

    assert_search_responses_equal(expected, mirrored_serial)

    var observed = execute_search_batch_with_prepared_snapshot(
        mirrored,
        requests,
        PreparedExactSearchBatchConfig(2, scoring_config),
    )
    assert_search_responses_equal(expected, observed)


def test_segment_mirror_batch_matches_serial_default() raises:
    var service_root = unique_service_root(
        "kayak-prepared-search-batch-segment-mirror-default"
    )
    build_service_fixture(service_root)

    assert_segment_mirror_batch_matches_serial(
        service_root,
        centroid_postings_imputed_search_plan(
            2,
            candidate_k(),
            best_effort_faithfulness_policy(),
        ),
        ExactScoringConfig(),
    )
    assert_segment_mirror_batch_matches_serial(
        service_root,
        centroid_postings_imputed_flat_search_plan(
            2,
            candidate_k(),
            best_effort_faithfulness_policy(),
        ),
        ExactScoringConfig(),
    )


def test_segment_mirror_batch_matches_serial_parallel_override() raises:
    var service_root = unique_service_root(
        "kayak-prepared-search-batch-segment-mirror-override"
    )
    build_service_fixture(service_root)
    var scoring_config = ExactScoringConfig()
    scoring_config.parallel_work_item_count_override = 2

    assert_segment_mirror_batch_matches_serial(
        service_root,
        centroid_postings_imputed_search_plan(
            2,
            candidate_k(),
            best_effort_faithfulness_policy(),
        ),
        scoring_config,
    )
    assert_segment_mirror_batch_matches_serial(
        service_root,
        centroid_postings_imputed_flat_search_plan(
            2,
            candidate_k(),
            best_effort_faithfulness_policy(),
        ),
        scoring_config,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
