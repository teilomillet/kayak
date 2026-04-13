from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    GemGraphBuildConfig,
    GemGraphIndex,
    GemGraphTrainingPair,
    build_gem_graph_index,
    build_gem_graph_index_with_config,
    pack_documents,
)
from kayak.index.gem_graph import inject_shortcuts


def neighbor_doc_indices_for(
    read index: GemGraphIndex, document_index: Int
) -> List[Int]:
    var neighbor_doc_indices = List[Int]()
    for neighbor_index in range(
        index.neighbor_offsets[document_index],
        index.neighbor_offsets[document_index + 1],
    ):
        neighbor_doc_indices.append(index.neighbor_doc_indices[neighbor_index])
    return neighbor_doc_indices^


def test_bridge_document_keeps_neighbors_from_both_clusters() raises:
    var index = build_gem_graph_index(
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
                EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
                EncodedDocument("doc-b", [[1.0, 0.0], [0.0, 1.0]]),
            ]
        ),
        2,
        2,
        2,
        1,
        2,
    )

    var bridge_neighbors = neighbor_doc_indices_for(index, 2)
    assert_equal(len(bridge_neighbors), 2)
    assert_equal(bridge_neighbors[0] == 0 or bridge_neighbors[1] == 0, True)
    assert_equal(bridge_neighbors[0] == 1 or bridge_neighbors[1] == 1, True)


def test_adaptive_cutoff_can_keep_more_clusters_than_fixed_cutoff() raises:
    var index = build_gem_graph_index_with_config(
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
                EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
            ]
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

    assert_equal(index.doc_profile_offsets[1] - index.doc_profile_offsets[0], 1)
    assert_equal(index.doc_profile_offsets[2] - index.doc_profile_offsets[1], 2)


def make_manual_shortcut_index() raises -> GemGraphIndex:
    return GemGraphIndex(
        ["doc-a", "doc-b", "doc-c"],
        [0, 1, 2, 3],
        [0, 1, 0],
        [1, 1, 1],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0]],
        [0, 0],
        [0, 1, 2, 3],
        [0, 0, 0],
        [1.0, 1.0, 1.0],
        [0, 3],
        [0, 1, 2],
        [0],
        [0, 1, 3, 4],
        [1, 0, 2, 1],
        2,
        0,
        1,
        False,
        0,
        1,
        2,
        False,
    )


def test_shortcut_injection_adds_missing_semantic_edge() raises:
    var index = make_manual_shortcut_index()
    var neighbor_ids_by_doc = List[List[Int]]()
    neighbor_ids_by_doc.append([1])
    neighbor_ids_by_doc.append([0, 2])
    neighbor_ids_by_doc.append([1])

    var shortcut_count = inject_shortcuts(
        index,
        neighbor_ids_by_doc,
        GemGraphBuildConfig(
            2,
            1,
            1,
            1,
            2,
            False,
            10,
            3,
            1,
            True,
            1,
            1,
            [GemGraphTrainingPair(EncodedQuery([[1.0, 0.0]]), "doc-c")],
        ),
    )

    assert_equal(shortcut_count, 1)
    assert_equal(neighbor_ids_by_doc[0][0] == 2 or neighbor_ids_by_doc[0][1] == 2, True)
    assert_equal(neighbor_ids_by_doc[2][0] == 0 or neighbor_ids_by_doc[2][1] == 0, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
