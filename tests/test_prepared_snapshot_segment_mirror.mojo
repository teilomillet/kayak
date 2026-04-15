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
    ExactCpuBackend,
    NamespaceId,
    SearchPlan,
    SearchRequest,
    SealedSegmentManifest,
    SnapshotId,
    TenantId,
    UpsertDocument,
    UpsertDocumentsRequest,
    VECTOR_SCALAR_NAME,
    best_effort_faithfulness_policy,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    create_collection,
    create_snapshot,
    execute_search_with_prepared_snapshot,
    match_all_filter,
    prepare_service_search_snapshot,
    upsert_documents,
)
from kayak.collections import (
    centroid_postings_search_artifact,
    load_sealed_segment_manifest,
    load_snapshot_manifest,
    save_sealed_segment_manifest,
)
from kayak.collections.paths import (
    collection_segment_root,
    collection_snapshot_root,
)
from kayak.service.paths import service_collection_root
from kayak.storage import (
    build_stored_centroid_posting_index,
    load_stored_packed_index,
    save_stored_centroid_posting_index,
)


def collection_id_fixture() raises -> CollectionId:
    return CollectionId("news")


def tenant_id_fixture() raises -> TenantId:
    return TenantId("tenant-a")


def namespace_id_fixture() raises -> NamespaceId:
    return NamespaceId("search")


def snapshot_id_fixture() raises -> SnapshotId:
    return SnapshotId("snapshot-0001")


def unique_service_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def basis_vector(dim_index: Int, scale: Float32 = 1.0) -> List[Float32]:
    var values = List[Float32]()
    values.reserve(128)

    for dim in range(128):
        if dim == dim_index:
            values.append(scale)
        else:
            values.append(0.0)

    return values^


def document_block(
    doc_id: String, start_dim: Int, scale: Float32
) raises -> EncodedDocument:
    var token_vectors = List[List[Float32]]()

    for token_index in range(32):
        token_vectors.append(basis_vector(start_dim + token_index, scale))

    return EncodedDocument(doc_id, token_vectors^)


def make_query32() raises -> EncodedQuery:
    var token_vectors = List[List[Float32]]()

    for token_index in range(32):
        token_vectors.append(basis_vector(token_index))

    return EncodedQuery(token_vectors^)


def build_dim128_service_fixture(service_root: Path) raises:
    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            collection_id_fixture(),
            tenant_id_fixture(),
            namespace_id_fixture(),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            128,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            collection_id_fixture(),
            tenant_id_fixture(),
            namespace_id_fixture(),
            [
                UpsertDocument(
                    document_block("doc-a", 0, 1.0),
                    "alpha document",
                ),
                UpsertDocument(
                    document_block("doc-b", 0, 0.9),
                    "beta document",
                ),
                UpsertDocument(
                    document_block("doc-c", 32, 1.0),
                    "gamma document",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            collection_id_fixture(),
            tenant_id_fixture(),
            namespace_id_fixture(),
            snapshot_id_fixture(),
            "publish centroid segment-mirror snapshot",
        ),
    )


def attach_centroid_postings_artifact(service_root: Path) raises:
    var collection_root = service_collection_root(
        service_root,
        tenant_id_fixture(),
        namespace_id_fixture(),
        collection_id_fixture(),
    )
    var snapshot = load_snapshot_manifest(
        collection_snapshot_root(collection_root, snapshot_id_fixture())
    )

    for segment_id in snapshot.segment_ids:
        var segment_root = collection_segment_root(collection_root, segment_id)
        var segment = load_sealed_segment_manifest(segment_root)
        var stored_index = load_stored_packed_index(
            segment_root / segment.packed_index_root
        )

        save_stored_centroid_posting_index(
            segment_root / "centroid_postings",
            build_stored_centroid_posting_index(stored_index.copy(), 0),
        )

        var search_artifacts = segment.search_artifacts.copy()
        var has_centroid_postings = False
        for artifact in search_artifacts:
            if artifact.family == "centroid_postings":
                has_centroid_postings = True
                break

        if not has_centroid_postings:
            search_artifacts.append(
                centroid_postings_search_artifact("centroid_postings")
            )

        save_sealed_segment_manifest(
            segment_root,
            SealedSegmentManifest(
                segment.segment_id,
                segment.collection_id,
                segment.tenant_id,
                segment.namespace_id,
                segment.generation,
                segment.model_name,
                segment.vector_scalar_name,
                segment.vector_dim,
                segment.packed_index_root,
                segment.document_representation_transforms,
                search_artifacts,
                segment.text_corpus_root,
                segment.stats,
            ),
        )


def assert_hit_lists_equal(
    read actual: List[CollectionHit], read expected: List[CollectionHit]
) raises:
    assert_equal(len(actual), len(expected))
    for index in range(len(actual)):
        assert_equal(actual[index].doc_id, expected[index].doc_id)
        assert_equal(actual[index].score, expected[index].score)


def assert_prepared_snapshot_segment_mirror_matches_baseline(
    service_root: Path, read plan: SearchPlan
) raises:
    var request = SearchRequest(
        collection_id_fixture(),
        tenant_id_fixture(),
        namespace_id_fixture(),
        snapshot_id_fixture(),
        make_query32(),
        "colbertv2",
        match_all_filter(),
        plan,
        False,
    )
    var baseline = prepare_service_search_snapshot(
        service_root,
        collection_id_fixture(),
        tenant_id_fixture(),
        namespace_id_fixture(),
        snapshot_id_fixture(),
        False,
        False,
    )
    var mirrored = prepare_service_search_snapshot(
        service_root,
        collection_id_fixture(),
        tenant_id_fixture(),
        namespace_id_fixture(),
        snapshot_id_fixture(),
        False,
        True,
    )
    var backend = ExactCpuBackend()

    assert_equal(mirrored.dim128_segment_mirrors_loaded, True)
    assert_equal(
        len(mirrored.dim128_segment_mirrors),
        len(mirrored.snapshot.segments),
    )

    var expected = execute_search_with_prepared_snapshot(
        backend,
        baseline,
        request,
    )
    var actual = execute_search_with_prepared_snapshot(
        backend,
        mirrored,
        request,
    )

    assert_hit_lists_equal(actual.hits, expected.hits)
    assert_equal(actual.hits[0].doc_id, "doc-a")
    assert_equal(actual.hits[1].doc_id, "doc-b")


def test_prepared_snapshot_segment_mirror_opt_in_matches_baseline() raises:
    var service_root = unique_service_root("kayak-prepared-segment-mirror")
    build_dim128_service_fixture(service_root)
    attach_centroid_postings_artifact(service_root)

    assert_prepared_snapshot_segment_mirror_matches_baseline(
        service_root,
        centroid_postings_imputed_search_plan(
            2,
            3,
            best_effort_faithfulness_policy(),
        ),
    )
    assert_prepared_snapshot_segment_mirror_matches_baseline(
        service_root,
        centroid_postings_imputed_flat_search_plan(
            2,
            3,
            best_effort_faithfulness_policy(),
        ),
    )


def test_prepare_service_search_snapshot_rejects_segment_mirrors_on_non_dim128_collection() raises:
    var service_root = unique_service_root("kayak-prepared-segment-mirror-nondim128")
    _ = create_collection(
        service_root,
        CreateCollectionRequest(
            collection_id_fixture(),
            tenant_id_fixture(),
            namespace_id_fixture(),
            "colbertv2",
            VECTOR_SCALAR_NAME,
            2,
        ),
    )
    _ = upsert_documents(
        service_root,
        UpsertDocumentsRequest(
            collection_id_fixture(),
            tenant_id_fixture(),
            namespace_id_fixture(),
            [
                UpsertDocument(
                    EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                    "alpha document",
                ),
            ],
        ),
    )
    _ = create_snapshot(
        service_root,
        CreateSnapshotRequest(
            collection_id_fixture(),
            tenant_id_fixture(),
            namespace_id_fixture(),
            snapshot_id_fixture(),
            "publish non-dim128 snapshot",
        ),
    )

    var raised = False
    try:
        _ = prepare_service_search_snapshot(
            service_root,
            collection_id_fixture(),
            tenant_id_fixture(),
            namespace_id_fixture(),
            snapshot_id_fixture(),
            False,
            True,
        )
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
