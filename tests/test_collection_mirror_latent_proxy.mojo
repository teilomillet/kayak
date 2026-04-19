from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CandidateBudget,
    CandidateGenerator,
    CollectionId,
    DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    MetricScalar,
    NamespaceId,
    SearchPlan,
    SnapshotId,
    TenantId,
    VECTOR_SCALAR_NAME,
    document_encoder_compression_config_value,
    ensure_one_segment_collection_mirror_with_latent_proxy,
    exact_late_interaction_reference_scoring_semantics,
    exact_late_interaction_stage2_reference_operator,
    explain_collection_search,
    load_resolved_collection_snapshot,
    loaded_segment_has_latent_proxy_index,
    memory_tokens_document_encoder_compression,
    none_stage3_verifier_operator,
    oracle_full_recall_required_faithfulness_policy,
    pack_documents,
)
from kayak.index import (
    LATENT_PROXY_ACTIVATION_RELU,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
    LatentProxyIndex,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
)
from kayak.storage import StoredLatentProxyIndex, StoredPackedIndex
from kayak.text import DocumentTextCorpus


def unique_collection_root(prefix: String) -> Path:
    var suffix = 0

    while True:
        var root = Path("/tmp/" + prefix + "-" + String(suffix))
        if not root.exists():
            return root
        suffix += 1


def mirror_fixture_index() raises -> StoredPackedIndex:
    return StoredPackedIndex(
        "dataset://latent-proxy",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
                EncodedDocument("doc-b", [[0.0, 1.0], [0.0, 1.0]]),
            ]
        ),
    )


def mirror_fixture_latent_proxy() raises -> StoredLatentProxyIndex:
    return StoredLatentProxyIndex(
        "",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        2,
        0,
        LatentQueryProjection(
            2,
            2,
            1.0,
            [
                LatentQueryProjectionBlock(
                    LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
                    LATENT_PROXY_ACTIVATION_RELU,
                    2,
                    2,
                    [[1.0, 0.0], [0.0, 1.0]],
                    [0.0, 0.0],
                    1.0,
                    False,
                    0.00001,
                    [],
                    [],
                )
            ],
        ),
        LatentProxyIndex(
            ["doc-a", "doc-b"],
            [[1.0, -1.0], [-1.0, 1.0]],
            2,
        ),
    )


def test_collection_mirror_with_latent_proxy_supports_native_stage1_search() raises:
    var document_encoder_compression = memory_tokens_document_encoder_compression(8)
    var collection_root = ensure_one_segment_collection_mirror_with_latent_proxy(
        unique_collection_root("kayak-collection-mirror-latent-proxy"),
        CollectionId("latent-proxy"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        mirror_fixture_index(),
        mirror_fixture_latent_proxy(),
        DocumentTextCorpus(
            ["doc-a", "doc-b"],
            ["alpha evidence", "beta evidence"],
        ),
        document_encoder_compression,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var explain = explain_collection_search(
        ExactCpuBackend(),
        EncodedQuery([[1.0, 0.0], [1.0, 0.0]]),
        snapshot,
        SearchPlan(
            CandidateGenerator("latent_proxy"),
            CandidateBudget(1, 2),
            oracle_full_recall_required_faithfulness_policy(),
            exact_late_interaction_reference_scoring_semantics(),
            exact_late_interaction_stage2_reference_operator(),
            none_stage3_verifier_operator(),
        ),
    )

    assert_equal(loaded_segment_has_latent_proxy_index(snapshot.segments[0]), True)
    assert_equal(snapshot.segments[0].has_text_corpus, True)
    assert_equal(
        snapshot.collection.document_encoder_compression.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    )
    assert_equal(
        document_encoder_compression_config_value(
            snapshot.collection.document_encoder_compression,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "8",
    )
    assert_equal(
        snapshot.segments[0].manifest.document_encoder_compression.kind,
        DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    )
    assert_equal(explain.final_hits[0].doc_id, "doc-a")
    assert_equal(explain.candidate_recall_at_final_k, MetricScalar(1.0))


def test_collection_mirror_with_latent_proxy_rejects_doc_id_mismatch() raises:
    var raised = False

    try:
        _ = ensure_one_segment_collection_mirror_with_latent_proxy(
            unique_collection_root("kayak-collection-mirror-latent-proxy-mismatch"),
            CollectionId("latent-proxy"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            mirror_fixture_index(),
            StoredLatentProxyIndex(
                "",
                "colbertv2",
                VECTOR_SCALAR_NAME,
                2,
                0,
                mirror_fixture_latent_proxy().query_projection,
                LatentProxyIndex(
                    ["doc-b", "doc-a"],
                    [[1.0, -1.0], [-1.0, 1.0]],
                    2,
                ),
            ),
        )
    except _:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
