from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    JudgedQuery,
    JudgedTask,
    SearchHit,
    evaluate_task,
    evaluate_task_with_verifier,
    exact_late_interaction_verifier,
    effective_candidate_k,
    no_verifier,
    pack_documents,
    rerank_hits_with_verifier,
    search_exact,
    search_exact_with_verifier,
)
from kayak.benchmarks import make_exact_search_fixture


def make_eval_fixture_task() raises -> JudgedTask:
    return JudgedTask(
        "mock",
        "verifier_eval",
        "Verifier evaluation fixture.",
        "mrr",
        2,
        1,
        1,
        2,
        [
            EncodedDocument("doc-a", [[1.0, 0.0]]),
            EncodedDocument("doc-b", [[0.75, 0.0]]),
            EncodedDocument("doc-c", [[0.0, 1.0]]),
            EncodedDocument("doc-d", [[0.0, 0.75]]),
        ],
        [
            JudgedQuery(
                "q-a",
                "first query",
                EncodedQuery([[1.0, 0.0]]),
                ["doc-a", "doc-b"],
            ),
            JudgedQuery(
                "q-b",
                "second query",
                EncodedQuery([[0.0, 1.0]]),
                ["doc-c", "doc-d"],
            ),
        ],
    )


def test_effective_candidate_k_never_drops_below_final_k() raises:
    assert_equal(effective_candidate_k(no_verifier(), 5), 5)
    assert_equal(
        effective_candidate_k(exact_late_interaction_verifier(2), 5), 5
    )
    assert_equal(
        effective_candidate_k(exact_late_interaction_verifier(7), 5), 7
    )


def test_noop_verifier_matches_exact_search() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 32, 5)
    var backend = ExactCpuBackend()

    var baseline_hits = search_exact(
        backend, fixture.query, fixture.index, fixture.top_k
    )
    var verifier_hits = search_exact_with_verifier(
        backend,
        fixture.query,
        fixture.index,
        fixture.top_k,
        no_verifier(),
    )

    assert_equal(len(baseline_hits), len(verifier_hits))
    for index in range(len(baseline_hits)):
        assert_equal(baseline_hits[index].doc_id, verifier_hits[index].doc_id)
        assert_equal(baseline_hits[index].score, verifier_hits[index].score)


def test_exact_verifier_can_reorder_candidate_window() raises:
    var query = EncodedQuery([[1.0, 0.0], [1.0, 0.0]])
    var index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
            EncodedDocument("doc-b", [[0.8, 0.0], [0.8, 0.0]]),
            EncodedDocument("doc-c", [[0.0, 1.0], [0.0, 1.0]]),
        ]
    )
    var candidate_hits = List[SearchHit]()
    candidate_hits.append(SearchHit("doc-b", 9.0))
    candidate_hits.append(SearchHit("doc-a", 8.0))
    candidate_hits.append(SearchHit("doc-c", 7.0))

    var reranked_hits = rerank_hits_with_verifier(
        query,
        index,
        candidate_hits,
        2,
        exact_late_interaction_verifier(3),
    )

    assert_equal(len(reranked_hits), 2)
    assert_equal(reranked_hits[0].doc_id, "doc-a")
    assert_equal(reranked_hits[1].doc_id, "doc-b")
    assert_equal(reranked_hits[0].score, 2.0)
    assert_equal(reranked_hits[1].score, 1.6)


def test_exact_verifier_matches_exact_search_when_stage_one_is_exact() raises:
    var fixture = make_exact_search_fixture(24, 12, 6, 32, 5)
    var backend = ExactCpuBackend()

    var baseline_hits = search_exact(
        backend, fixture.query, fixture.index, fixture.top_k
    )
    var verifier_hits = search_exact_with_verifier(
        backend,
        fixture.query,
        fixture.index,
        fixture.top_k,
        exact_late_interaction_verifier(fixture.top_k + 4),
    )

    assert_equal(len(baseline_hits), len(verifier_hits))
    for index in range(len(baseline_hits)):
        assert_equal(baseline_hits[index].doc_id, verifier_hits[index].doc_id)
        assert_equal(baseline_hits[index].score, verifier_hits[index].score)


def test_evaluate_task_with_verifier_matches_exact_evaluation() raises:
    var backend = ExactCpuBackend()
    var task = make_eval_fixture_task()

    var baseline = evaluate_task(backend, task)
    var verifier_eval = evaluate_task_with_verifier(
        backend, task, exact_late_interaction_verifier(4)
    )

    assert_equal(baseline.primary_metric, verifier_eval.primary_metric)
    assert_equal(baseline.primary_value, verifier_eval.primary_value)
    assert_equal(
        baseline.mean_reciprocal_rank, verifier_eval.mean_reciprocal_rank
    )
    assert_equal(baseline.mean_recall_at_k, verifier_eval.mean_recall_at_k)
    assert_equal(baseline.success_rate_at_k, verifier_eval.success_rate_at_k)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
