from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    GEM_GRAPH_ADAPTIVE_LABEL_POLICY_RELEVANT_CLUSTER_COVERAGE,
    GemGraphBuildConfig,
    GemGraphIndex,
    GemGraphTrainingPair,
    build_gem_graph_index,
    build_gem_graph_index_with_config,
    pack_documents,
)
from kayak.index.gem_graph import (
    QuantizedCodeHistogram,
    adaptive_profile_label,
    build_document_quantized_histogram,
    build_quantization_distance_matrix,
    inject_shortcuts,
    select_neighbors_by_cluster_heuristic,
)


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


def test_cluster_entry_uses_first_cluster_member_like_reference() raises:
    var index = build_gem_graph_index(
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0]]),
                EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            ]
        ),
        1,
        1,
        1,
        1,
        2,
    )

    assert_equal(index.cluster_count, 1)
    assert_equal(index.entry_doc_indices[0], 0)


def test_cluster_heuristic_matches_reference_diversification_rule() raises:
    var histograms_by_doc = List[QuantizedCodeHistogram]()
    histograms_by_doc.append(QuantizedCodeHistogram([0], [1]))
    histograms_by_doc.append(QuantizedCodeHistogram([1], [1]))
    histograms_by_doc.append(QuantizedCodeHistogram([1], [1]))
    histograms_by_doc.append(QuantizedCodeHistogram([2], [1]))
    var histogram_totals_by_doc = [1, 1, 1, 1]
    var distance_matrix = [
        0.0,
        1.0,
        2.0,
        1.0,
        0.0,
        2.0,
        2.0,
        2.0,
        0.0,
    ]

    var selected = select_neighbors_by_cluster_heuristic(
        0,
        [1, 2, 3],
        2,
        histograms_by_doc,
        histogram_totals_by_doc,
        distance_matrix,
        3,
    )

    assert_equal(len(selected), 2)
    assert_equal(selected[0] == 1 or selected[1] == 1, True)
    assert_equal(selected[0] == 3 or selected[1] == 3, True)


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


def test_adaptive_label_policy_can_require_more_than_first_hit() raises:
    var label_first_hit = adaptive_profile_label(
        EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        1,
        2,
        GEM_GRAPH_ADAPTIVE_LABEL_POLICY_FIRST_RELEVANT_CLUSTER_RANK,
    )
    var label_coverage = adaptive_profile_label(
        EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        1,
        2,
        GEM_GRAPH_ADAPTIVE_LABEL_POLICY_RELEVANT_CLUSTER_COVERAGE,
    )

    assert_equal(label_first_hit, 1)
    assert_equal(label_coverage, 2)


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
        1,
    )


def manual_shortcut_histograms(
    read index: GemGraphIndex
) raises -> List[QuantizedCodeHistogram]:
    var histograms = List[QuantizedCodeHistogram]()
    for document_index in range(index.document_count):
        histograms.append(
            build_document_quantized_histogram(
                index.doc_code_ids,
                index.doc_code_counts,
                index.doc_code_offsets[document_index],
                index.doc_code_offsets[document_index + 1],
            )
        )
    return histograms^


def manual_shortcut_histogram_totals(
    read histograms: List[QuantizedCodeHistogram]
) -> List[Int]:
    var totals = List[Int]()
    for histogram in histograms:
        var total = 0
        for count in histogram.counts:
            total += count
        totals.append(total)
    return totals^


def test_shortcut_injection_adds_missing_semantic_edge() raises:
    var index = make_manual_shortcut_index()
    var histograms = manual_shortcut_histograms(index)
    var histogram_totals = manual_shortcut_histogram_totals(histograms)
    var distance_matrix = build_quantization_distance_matrix(
        index.quantization_centroids
    )
    var neighbor_ids_by_doc = List[List[Int]]()
    neighbor_ids_by_doc.append([1])
    neighbor_ids_by_doc.append([0, 2])
    neighbor_ids_by_doc.append([1])

    var shortcut_count = inject_shortcuts(
        index,
        neighbor_ids_by_doc,
        histograms,
        histogram_totals,
        distance_matrix,
        len(index.quantization_centroids),
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


def test_shortcut_injection_can_rewire_saturated_vertices() raises:
    var index = make_manual_shortcut_index()
    var histograms = manual_shortcut_histograms(index)
    var histogram_totals = manual_shortcut_histogram_totals(histograms)
    var distance_matrix = build_quantization_distance_matrix(
        index.quantization_centroids
    )
    var neighbor_ids_by_doc = List[List[Int]]()
    neighbor_ids_by_doc.append([1])
    neighbor_ids_by_doc.append([0])
    neighbor_ids_by_doc.append([1])

    var shortcut_count = inject_shortcuts(
        index,
        neighbor_ids_by_doc,
        histograms,
        histogram_totals,
        distance_matrix,
        len(index.quantization_centroids),
        GemGraphBuildConfig(
            2,
            1,
            1,
            1,
            1,
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
    assert_equal(len(neighbor_ids_by_doc[0]), 1)
    assert_equal(len(neighbor_ids_by_doc[2]), 1)
    assert_equal(neighbor_ids_by_doc[0][0], 2)
    assert_equal(neighbor_ids_by_doc[2][0], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
