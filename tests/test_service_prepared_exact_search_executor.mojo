from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionHit,
    CollectionId,
    collection_segment_root,
    collection_snapshot_root,
    CreateCollectionRequest,
    CreateSnapshotRequest,
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    ExactScoringConfig,
    NamespaceId,
    SnapshotId,
    TenantId,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    default_exact_search_request,
    execute_search_with_prepared_snapshot,
    hybrid_flat_dim128_default_root,
    load_snapshot_manifest,
    prepare_service_exact_search_executor,
    prepare_service_exact_search_executor_with_config,
    prepare_service_search_snapshot,
    create_collection,
    create_snapshot,
    upsert_documents,
)
from kayak.service.paths import service_collection_root


def unique_service_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def make_query() raises -> EncodedQuery:
    return EncodedQuery(
        [
            make_basis_vector(1.0, 0.0),
            make_basis_vector(0.0, 1.0),
        ]
    )


def make_basis_vector(first: Float32, second: Float32) -> List[Float32]:
    var values = List[Float32]()
    values.reserve(128)
    values.append(first)
    values.append(second)
    for _ in range(2, 128):
        values.append(0.0)
    return values^


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
                        [
                            make_basis_vector(1.0, 0.0),
                            make_basis_vector(0.0, 1.0),
                        ],
                    ),
                    "alpha document",
                ),
                UpsertDocument(
                    make_document(
                        "doc-b",
                        [
                            make_basis_vector(0.8, 0.0),
                            make_basis_vector(0.0, 0.8),
                        ],
                    ),
                    "beta document",
                ),
                UpsertDocument(
                    make_document(
                        "doc-c",
                        [
                            make_basis_vector(0.5, 0.0),
                            make_basis_vector(0.0, 0.5),
                        ],
                    ),
                    "gamma document",
                ),
                UpsertDocument(
                    make_document(
                        "doc-d",
                        [
                            make_basis_vector(0.1, 0.0),
                            make_basis_vector(0.0, 0.1),
                        ],
                    ),
                    "delta document",
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


def assert_hit_doc_ids_equal(
    actual: List[CollectionHit], expected: List[CollectionHit]
) raises:
    assert_equal(len(actual), len(expected))
    for index in range(len(actual)):
        assert_equal(actual[index].doc_id, expected[index].doc_id)
        assert_equal(actual[index].score, expected[index].score)


def test_prepared_exact_search_executor_matches_prepared_snapshot_runtime() raises:
    var service_root = unique_service_root("kayak-service-prepared-executor-default")
    build_service_fixture(service_root)
    var expected_snapshot = prepare_service_search_snapshot(
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
        4,
        "colbertv2",
    )

    var expected = execute_search_with_prepared_snapshot(
        ExactCpuBackend(),
        expected_snapshot,
        request,
    )
    var executor = prepare_service_exact_search_executor(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        False,
    )
    var actual = executor.execute_search(request)

    assert_equal(actual.snapshot_id.value, expected.snapshot_id.value)
    assert_hit_doc_ids_equal(actual.hits, expected.hits)

    var collection_root = service_collection_root(
        service_root,
        TenantId("tenant-a"),
        NamespaceId("search"),
        CollectionId("news"),
    )
    var snapshot = load_snapshot_manifest(
        collection_snapshot_root(collection_root, SnapshotId("snapshot-0001"))
    )
    assert_equal(len(snapshot.segment_ids), 1)
    var hybrid_root = hybrid_flat_dim128_default_root(
        collection_segment_root(collection_root, snapshot.segment_ids[0])
    )
    assert_equal((hybrid_root / "manifest.tsv").exists(), True)
    assert_equal((hybrid_root / "token_values.bin").exists(), True)


def test_prepared_exact_search_executor_parallel_override_keeps_exact_hits_identical() raises:
    var service_root = unique_service_root("kayak-service-prepared-executor-config")
    build_service_fixture(service_root)
    var request = default_exact_search_request(
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        make_query(),
        4,
        "colbertv2",
    )

    var parallel_config = ExactScoringConfig()
    parallel_config.parallel_work_item_count_override = 2
    var parallel_executor = prepare_service_exact_search_executor_with_config(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        parallel_config,
        False,
    )

    var serial_config = ExactScoringConfig()
    serial_config.enable_parallel_scoring = False
    var serial_executor = prepare_service_exact_search_executor_with_config(
        service_root,
        CollectionId("news"),
        TenantId("tenant-a"),
        NamespaceId("search"),
        SnapshotId("snapshot-0001"),
        serial_config,
        False,
    )

    var parallel = parallel_executor.execute_search(request)
    var serial = serial_executor.execute_search(request)

    assert_hit_doc_ids_equal(parallel.hits, serial.hits)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
