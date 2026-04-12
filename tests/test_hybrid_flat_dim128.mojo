from std.testing import TestSuite, assert_equal

from kayak import (
    FlatQueryDim128,
    HybridFlatDim128Index,
    StoredHybridFlatDim128Index,
    VECTOR_SCALAR_NAME,
    build_flat_query_dim128,
    build_hybrid_flat_dim128_index,
    exact_scores_for_hybrid_flat_index_dim128,
    exact_scores_for_hybrid_flat_index_dim128_with_flat_query,
    load_stored_hybrid_flat_dim128_index,
    save_stored_hybrid_flat_dim128_index,
    search_exact_hybrid_flat_dim128,
    search_exact_hybrid_flat_dim128_with_flat_query,
)
from kayak.benchmarks import make_exact_search_fixture
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from std.pathlib import Path


def assert_hybrid_indexes_equal(
    read lhs: HybridFlatDim128Index, read rhs: HybridFlatDim128Index
) raises:
    assert_equal(lhs.document_count, rhs.document_count)
    assert_equal(lhs.total_vector_count, rhs.total_vector_count)
    assert_equal(lhs.vector_dim, rhs.vector_dim)
    assert_equal(len(lhs.doc_ids), len(rhs.doc_ids))
    assert_equal(len(lhs.doc_offsets), len(rhs.doc_offsets))
    assert_equal(len(lhs.token_values), len(rhs.token_values))

    for index in range(len(lhs.doc_ids)):
        assert_equal(lhs.doc_ids[index], rhs.doc_ids[index])

    for index in range(len(lhs.doc_offsets)):
        assert_equal(lhs.doc_offsets[index], rhs.doc_offsets[index])

    for index in range(len(lhs.token_values)):
        assert_equal(lhs.token_values[index], rhs.token_values[index])


def assert_flat_queries_equal(
    read lhs: FlatQueryDim128, read rhs: FlatQueryDim128
) raises:
    assert_equal(lhs.vector_dim, rhs.vector_dim)
    assert_equal(lhs.vector_count, rhs.vector_count)
    assert_equal(len(lhs.token_values), len(rhs.token_values))

    for index in range(len(lhs.token_values)):
        assert_equal(lhs.token_values[index], rhs.token_values[index])


def test_hybrid_flat_dim128_scores_match_default_backend() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 128, 5)
    var backend = ExactCpuBackend()
    var hybrid_index = build_hybrid_flat_dim128_index(fixture.index)

    var nested_scores = backend.score_all(fixture.query, fixture.index)
    var hybrid_scores = exact_scores_for_hybrid_flat_index_dim128(
        fixture.query,
        fixture.index,
        hybrid_index,
        backend.scoring_config,
    )

    assert_equal(len(nested_scores), len(hybrid_scores))
    for index in range(len(nested_scores)):
        assert_equal(nested_scores[index], hybrid_scores[index])


def test_hybrid_flat_dim128_hits_match_default_search() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 128, 5)
    var backend = ExactCpuBackend()
    var hybrid_index = build_hybrid_flat_dim128_index(fixture.index)

    var nested_hits = search_exact(backend, fixture.query, fixture.index, fixture.top_k)
    var hybrid_hits = search_exact_hybrid_flat_dim128(
        fixture.query,
        fixture.index,
        hybrid_index,
        fixture.top_k,
        backend.scoring_config,
    )

    assert_equal(len(nested_hits), len(hybrid_hits))
    for index in range(len(nested_hits)):
        assert_equal(nested_hits[index].doc_id, hybrid_hits[index].doc_id)
        assert_equal(nested_hits[index].score, hybrid_hits[index].score)


def test_flat_query_dim128_scores_match_nested_query_path() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 128, 5)
    var backend = ExactCpuBackend()
    var hybrid_index = build_hybrid_flat_dim128_index(fixture.index)
    var flat_query = build_flat_query_dim128(fixture.query)

    var hybrid_scores = exact_scores_for_hybrid_flat_index_dim128(
        fixture.query,
        fixture.index,
        hybrid_index,
        backend.scoring_config,
    )
    var flat_query_scores = exact_scores_for_hybrid_flat_index_dim128_with_flat_query(
        flat_query,
        fixture.index,
        hybrid_index,
        backend.scoring_config,
    )

    assert_equal(len(hybrid_scores), len(flat_query_scores))
    for index in range(len(hybrid_scores)):
        assert_equal(hybrid_scores[index], flat_query_scores[index])


def test_flat_query_dim128_hits_match_nested_query_path() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 128, 5)
    var backend = ExactCpuBackend()
    var hybrid_index = build_hybrid_flat_dim128_index(fixture.index)
    var flat_query = build_flat_query_dim128(fixture.query)

    var hybrid_hits = search_exact_hybrid_flat_dim128(
        fixture.query,
        fixture.index,
        hybrid_index,
        fixture.top_k,
        backend.scoring_config,
    )
    var flat_query_hits = search_exact_hybrid_flat_dim128_with_flat_query(
        flat_query,
        fixture.index,
        hybrid_index,
        fixture.top_k,
        backend.scoring_config,
    )

    assert_equal(len(hybrid_hits), len(flat_query_hits))
    for index in range(len(hybrid_hits)):
        assert_equal(hybrid_hits[index].doc_id, flat_query_hits[index].doc_id)
        assert_equal(hybrid_hits[index].score, flat_query_hits[index].score)


def test_hybrid_flat_dim128_storage_roundtrip_preserves_layout() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 128, 5)
    var root = Path("/tmp/kayak-hybrid-flat-dim128-roundtrip")
    var expected = StoredHybridFlatDim128Index(
        "mock://hybrid-roundtrip",
        "mock-model",
        VECTOR_SCALAR_NAME,
        build_hybrid_flat_dim128_index(fixture.index),
    )

    save_stored_hybrid_flat_dim128_index(root, expected)
    var loaded = load_stored_hybrid_flat_dim128_index(root)

    assert_equal(loaded.dataset_id, "mock://hybrid-roundtrip")
    assert_equal(loaded.model_name, "mock-model")
    assert_equal(loaded.vector_scalar_name, VECTOR_SCALAR_NAME)
    assert_hybrid_indexes_equal(expected.index, loaded.index)


def test_build_flat_query_dim128_preserves_query_shape() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 128, 5)
    var flat_query = build_flat_query_dim128(fixture.query)
    var rebuilt_flat_query = build_flat_query_dim128(fixture.query)

    assert_flat_queries_equal(flat_query, rebuilt_flat_query)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
