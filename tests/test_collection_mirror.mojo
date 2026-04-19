from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    CollectionId,
    DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
    EncodedDocument,
    EncodedQuery,
    GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    GemGraphBuildConfig,
    GemGraphTrainingPair,
    NamespaceId,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    document_encoder_compression_config_value,
    ensure_one_segment_collection_mirror,
    loaded_segment_has_gem_graph_index,
    loaded_segment_stored_gem_graph_index,
    load_resolved_collection_snapshot,
    memory_tokens_document_encoder_compression,
    pack_documents,
)
from kayak.numeric import VECTOR_SCALAR_NAME
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
        "mock://mirror",
        "colbertv2",
        VECTOR_SCALAR_NAME,
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                EncodedDocument("doc-b", [[0.0, 1.0], [1.0, 0.0]]),
            ]
        ),
    )


def test_collection_mirror_persists_optional_text_corpus() raises:
    var stored_index = mirror_fixture_index()
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-collection-mirror-text"),
        CollectionId("mirror-text"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        stored_index,
        DocumentTextCorpus(
            ["doc-a", "doc-b"],
            ["alpha clause evidence", "beta supporting passage"],
        ),
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )

    assert_equal(snapshot.segments[0].has_text_corpus, True)
    assert_equal(
        snapshot.segments[0].stored_text_corpus.corpus.doc_ids[1],
        "doc-b",
    )
    assert_equal(
        snapshot.segments[0].stored_text_corpus.corpus.texts[0],
        "alpha clause evidence",
    )


def test_collection_mirror_rejects_misaligned_text_corpus_doc_order() raises:
    var raised = False

    try:
        _ = ensure_one_segment_collection_mirror(
            unique_collection_root("kayak-collection-mirror-misaligned"),
            CollectionId("mirror-text"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            mirror_fixture_index(),
            DocumentTextCorpus(
                ["doc-b", "doc-a"],
                ["beta supporting passage", "alpha clause evidence"],
            ),
        )
    except _:
        raised = True

    assert_equal(raised, True)


def test_collection_mirror_preserves_document_encoder_compression_provenance() raises:
    var document_encoder_compression = memory_tokens_document_encoder_compression(8)
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-collection-mirror-encoder-compression"),
        CollectionId("mirror-compression"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        mirror_fixture_index(),
        DocumentTextCorpus(
            ["doc-a", "doc-b"],
            ["alpha clause evidence", "beta supporting passage"],
        ),
        document_encoder_compression,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )

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
    assert_equal(
        document_encoder_compression_config_value(
            snapshot.segments[0].manifest.document_encoder_compression,
            DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
        ),
        "8",
    )


def test_collection_mirror_can_build_configured_gem_graph_sidecar() raises:
    var collection_root = ensure_one_segment_collection_mirror(
        unique_collection_root("kayak-collection-mirror-gem-config"),
        CollectionId("mirror-gem-config"),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        mirror_fixture_index(),
        DocumentTextCorpus(
            ["doc-a", "doc-b"],
            ["alpha clause evidence", "beta supporting passage"],
        ),
        GemGraphBuildConfig(
            2,
            2,
            1,
            2,
            2,
            True,
            2,
            2,
            1,
            True,
            1,
            1,
            [GemGraphTrainingPair(EncodedQuery([[1.0, 0.0]]), "doc-a")],
        ),
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )

    assert_equal(loaded_segment_has_gem_graph_index(snapshot.segments[0]), True)
    var stored_gem_graph = loaded_segment_stored_gem_graph_index(
        snapshot.segments[0]
    )
    assert_equal(stored_gem_graph.adaptive_cluster_cutoff_enabled, True)
    assert_equal(stored_gem_graph.adaptive_cluster_cutoff_max, 2)
    assert_equal(
        stored_gem_graph.adaptive_label_policy,
        GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    )
    assert_equal(stored_gem_graph.shortcut_candidate_k, 1)
    assert_equal(stored_gem_graph.shortcuts_enabled, True)
    assert_equal(stored_gem_graph.index.adaptive_cluster_cutoff_enabled, True)
    assert_equal(
        stored_gem_graph.index.adaptive_label_policy,
        GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    )
    assert_equal(stored_gem_graph.index.shortcut_candidate_k, 1)
    assert_equal(stored_gem_graph.index.shortcuts_enabled, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
