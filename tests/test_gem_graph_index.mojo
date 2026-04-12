from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    build_gem_graph_index,
    build_quantization_distance_matrix,
    document_profile_intersects_clusters,
    pack_documents,
    quantize_query_codes,
    quantized_chamfer_distance_for_document,
    query_entry_doc_indices,
    query_relevant_cluster_ids,
)


def test_gem_graph_index_builds_profiles_entries_and_neighbors() raises:
    var index = build_gem_graph_index(
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0]]),
                EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
            ]
        ),
        2,
        2,
        2,
        2,
        2,
    )

    assert_equal(index.document_count, 3)
    assert_equal(len(index.doc_code_offsets), 4)
    assert_equal(len(index.doc_profile_offsets), 4)
    assert_equal(len(index.neighbor_offsets), 4)
    assert_equal(index.quantization_centroid_count, 2)
    assert_equal(index.cluster_count, 2)
    assert_equal(index.graph_edge_count > 0, True)

    for document_index in range(index.document_count):
        assert_equal(
            index.doc_profile_offsets[document_index + 1]
            > index.doc_profile_offsets[document_index],
            True,
        )

    for cluster_index in range(index.cluster_count):
        if index.entry_doc_indices[cluster_index] == -1:
            assert_equal(
                index.cluster_offsets[cluster_index + 1]
                == index.cluster_offsets[cluster_index],
                True,
            )
        else:
            assert_equal(index.entry_doc_indices[cluster_index] >= 0, True)


def test_gem_graph_query_helpers_follow_cluster_filtered_path() raises:
    var index = build_gem_graph_index(
        pack_documents(
            [
                EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
                EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0]]),
                EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
            ]
        ),
        2,
        2,
        2,
        2,
        2,
    )
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var relevant_clusters = query_relevant_cluster_ids(query, index, 1)
    var entry_docs = query_entry_doc_indices(index, relevant_clusters)
    var query_codes = quantize_query_codes(query, index)
    var distance_matrix = build_quantization_distance_matrix(
        index.quantization_centroids
    )

    assert_equal(len(relevant_clusters) > 0, True)
    assert_equal(len(entry_docs) > 0, True)
    assert_equal(len(query_codes), 2)
    assert_equal(
        document_profile_intersects_clusters(index, entry_docs[0], relevant_clusters),
        True,
    )
    assert_equal(
        quantized_chamfer_distance_for_document(
            query_codes, index, entry_docs[0], distance_matrix
        )
        >= 0.0,
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
