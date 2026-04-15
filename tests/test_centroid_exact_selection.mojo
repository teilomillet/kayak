from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import FlatQueryDim128, ScoreScalar, VectorScalar
from kayak.index import CentroidPostingIndex
from kayak.planning.centroid_primitives import ScoredCentroidSelection
from kayak.planning.centroid_postings_flat_stage import (
    top_centroid_selection_for_flat_query_token_dim128,
    top_centroid_selection_for_flat_query_token_generic,
)
from kayak.planning.centroid_postings_stage import top_centroid_selection_for_query_token
from kayak.scoring.dot import dot_product


def assert_scored_centroid_selection_equal(
    read lhs: ScoredCentroidSelection,
    read rhs: ScoredCentroidSelection,
) raises:
    assert_equal(lhs.baseline_correction, rhs.baseline_correction)
    assert_equal(lhs.centroid_indices, rhs.centroid_indices)
    assert_equal(lhs.centroid_scores, rhs.centroid_scores)


def insert_reference_top2_centroid_match(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    centroid_index: Int,
    centroid_score: ScoreScalar,
):
    var insert_at = 0
    while (
        insert_at < len(centroid_scores)
        and centroid_scores[insert_at] >= centroid_score
    ):
        insert_at += 1

    if insert_at >= 2:
        if len(centroid_scores) < 2:
            centroid_indices.append(centroid_index)
            centroid_scores.append(centroid_score)
        return

    if len(centroid_scores) < 2:
        centroid_indices.append(centroid_index)
        centroid_scores.append(centroid_score)

    var current = len(centroid_scores) - 1
    while current > insert_at:
        centroid_indices[current] = centroid_indices[current - 1]
        centroid_scores[current] = centroid_scores[current - 1]
        current -= 1

    centroid_indices[insert_at] = centroid_index
    centroid_scores[insert_at] = centroid_score


def reference_exact_centroid_selection(
    read query_token: List[VectorScalar],
    read index: CentroidPostingIndex,
) -> ScoredCentroidSelection:
    var centroid_indices = List[Int]()
    var centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        insert_reference_top2_centroid_match(
            centroid_indices,
            centroid_scores,
            centroid_index,
            dot_product(query_token, index.centroid_vectors[centroid_index]),
        )

    return ScoredCentroidSelection(centroid_indices^, centroid_scores^, ScoreScalar(0.0))


def constant_vector(vector_dim: Int, value: Float64) -> List[VectorScalar]:
    var values = List[VectorScalar]()
    for _ in range(vector_dim):
        values.append(VectorScalar(value))
    return values^


def centroid_index_from_scalar_values(
    read centroid_values: List[Float64], vector_dim: Int
) raises -> CentroidPostingIndex:
    var centroid_dims = List[Int]()
    var centroid_vectors = List[List[VectorScalar]]()
    var posting_offsets = List[Int]()
    var posting_doc_indices = List[Int]()
    var posting_weights = List[Int]()

    posting_offsets.append(0)
    for value in centroid_values:
        centroid_dims.append(0)
        centroid_vectors.append(constant_vector(vector_dim, value))
        posting_doc_indices.append(0)
        posting_weights.append(1)
        posting_offsets.append(len(posting_doc_indices))

    return CentroidPostingIndex(
        centroid_dims^,
        centroid_vectors^,
        posting_offsets^,
        posting_doc_indices^,
        posting_weights^,
        vector_dim,
        1,
    )


def test_exact_nested_selector_keeps_earlier_equal_tail_entry() raises:
    var centroid_values = List[Float64]()
    centroid_values.append(1.0)
    centroid_values.append(1.0)
    centroid_values.append(1.0)
    var index = centroid_index_from_scalar_values(centroid_values, 1)
    var query_token = constant_vector(1, 1.0)

    assert_scored_centroid_selection_equal(
        top_centroid_selection_for_query_token(query_token, index),
        reference_exact_centroid_selection(query_token, index),
    )


def test_exact_nested_selector_replaces_weaker_second_with_equal_best() raises:
    var centroid_values = List[Float64]()
    centroid_values.append(2.0)
    centroid_values.append(1.0)
    centroid_values.append(2.0)
    var index = centroid_index_from_scalar_values(centroid_values, 1)
    var query_token = constant_vector(1, 1.0)

    assert_scored_centroid_selection_equal(
        top_centroid_selection_for_query_token(query_token, index),
        reference_exact_centroid_selection(query_token, index),
    )


def test_exact_flat_generic_selector_matches_reference_top2() raises:
    var centroid_values = List[Float64]()
    centroid_values.append(2.0)
    centroid_values.append(1.0)
    centroid_values.append(2.0)
    var index = centroid_index_from_scalar_values(centroid_values, 1)
    var query_token = constant_vector(1, 1.0)

    assert_scored_centroid_selection_equal(
        top_centroid_selection_for_flat_query_token_generic(query_token, 0, index),
        reference_exact_centroid_selection(query_token, index),
    )


def test_exact_flat_dim128_selector_matches_reference_top2() raises:
    var centroid_values = List[Float64]()
    centroid_values.append(2.0)
    centroid_values.append(1.0)
    centroid_values.append(2.0)
    var index = centroid_index_from_scalar_values(centroid_values, 128)
    var query_token = constant_vector(128, 1.0)
    var flat_query = FlatQueryDim128(query_token.copy(), 128)

    assert_scored_centroid_selection_equal(
        top_centroid_selection_for_flat_query_token_dim128(flat_query, 0, index),
        reference_exact_centroid_selection(query_token, index),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
