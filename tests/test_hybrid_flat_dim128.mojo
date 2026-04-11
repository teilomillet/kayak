from std.testing import TestSuite, assert_equal

from kayak.benchmarks import make_exact_search_fixture
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from benchmarks.structural.hybrid_dim128 import (
    build_hybrid_flat_dim128_index,
    exact_scores_for_hybrid_flat_index_dim128,
    search_exact_hybrid_flat_dim128,
)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
