from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    GemGraphBuildConfig,
    GemGraphTrainingPair,
    VECTOR_SCALAR_NAME,
    pack_documents,
)
from kayak.storage import (
    StoredPackedIndex,
    build_stored_gem_graph_index,
    build_stored_gem_graph_index_with_config,
    gem_graph_index_exists,
    load_stored_gem_graph_index,
    save_stored_gem_graph_index,
)


def test_gem_graph_store_roundtrip_preserves_metadata() raises:
    var root = Path("/tmp/kayak-gem-graph-store-roundtrip")
    var packed_index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[0.9, 0.1], [0.8, 0.2]]),
            EncodedDocument("doc-c", [[0.0, 1.0], [0.1, 0.9]]),
        ]
    )
    var stored = build_stored_gem_graph_index(
        StoredPackedIndex(
            "collection://news",
            "colbertv2",
            VECTOR_SCALAR_NAME,
            packed_index.copy(),
        ),
        2,
        2,
        1,
        2,
        3,
    )

    save_stored_gem_graph_index(
        root,
        stored.copy(),
    )

    var loaded = load_stored_gem_graph_index(root)

    assert_equal(gem_graph_index_exists(root), True)
    assert_equal(loaded.dataset_id, "collection://news")
    assert_equal(loaded.document_count, 3)
    assert_equal(loaded.cluster_count > 0, True)
    assert_equal(loaded.graph_edge_count > 0, True)
    assert_equal(loaded.shortcut_edge_count, 0)
    assert_equal(loaded.entry_point_count > 0, True)
    assert_equal(loaded.quantization_centroid_count, 2)
    assert_equal(loaded.index.doc_ids[0], "doc-a")
    assert_equal(loaded.index.document_count, 3)
    assert_equal(loaded.adaptive_cluster_cutoff_enabled, False)
    assert_equal(loaded.adaptive_cluster_cutoff_max, 0)
    assert_equal(loaded.shortcuts_enabled, False)
    assert_equal(loaded.artifact_byte_size > 0, True)


def test_gem_graph_store_roundtrip_preserves_adaptive_metadata() raises:
    var root = Path("/tmp/kayak-gem-graph-store-adaptive-roundtrip")
    var packed_index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
            EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
        ]
    )
    var stored = build_stored_gem_graph_index_with_config(
        StoredPackedIndex(
            "collection://adaptive",
            "colbertv2",
            VECTOR_SCALAR_NAME,
            packed_index.copy(),
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
            False,
            2,
            4,
            [
                GemGraphTrainingPair(EncodedQuery([[1.0, 0.0]]), "doc-a"),
                GemGraphTrainingPair(EncodedQuery([[0.0, 1.0]]), "doc-b"),
            ],
        ),
    )

    save_stored_gem_graph_index(root, stored.copy())
    var loaded = load_stored_gem_graph_index(root)

    assert_equal(loaded.adaptive_cluster_cutoff_enabled, True)
    assert_equal(loaded.adaptive_cluster_cutoff_max, 2)
    assert_equal(loaded.shortcuts_enabled, False)
    assert_equal(loaded.index.adaptive_cluster_cutoff_enabled, True)
    assert_equal(loaded.index.adaptive_cluster_cutoff_max, 2)
    assert_equal(loaded.index.shortcuts_enabled, False)
    assert_equal(loaded.index.doc_profile_offsets[2] - loaded.index.doc_profile_offsets[1], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
