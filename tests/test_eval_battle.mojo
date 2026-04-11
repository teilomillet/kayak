from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    JudgedQuery,
    JudgedTask,
    MetricScalar,
    ScoreScalar,
    SearchHit,
)
from kayak.eval import evaluate_task, recall_at_k, reciprocal_rank_at_k, success_at_k


struct Lcg(Copyable):
    var state: Int

    def __init__(out self, seed: Int):
        if seed == 0:
            self.state = 1
        else:
            self.state = seed

    def next_int(mut self, limit: Int) -> Int:
        self.state = (self.state * 48271) % 2147483647
        return self.state % limit


def make_random_hits(mut rng: Lcg, hit_count: Int) -> List[SearchHit]:
    var hits = List[SearchHit]()

    for index in range(hit_count):
        hits.append(
            SearchHit(
                "doc-" + String(index), ScoreScalar(rng.next_int(7))
            )
        )

    return hits^


def make_random_relevant_doc_ids(
    mut rng: Lcg, hit_count: Int
) -> List[String]:
    var relevant_doc_ids = List[String]()

    for index in range(hit_count):
        if rng.next_int(2) == 1:
            relevant_doc_ids.append("doc-" + String(index))

    return relevant_doc_ids^


def reference_is_relevant(
    doc_id: String, relevant_doc_ids: List[String]
) -> Bool:
    for relevant_doc_id in relevant_doc_ids:
        if relevant_doc_id == doc_id:
            return True

    return False


def reference_success_at_k(
    hits: List[SearchHit], relevant_doc_ids: List[String], k: Int
) -> MetricScalar:
    var limit = k
    if limit > len(hits):
        limit = len(hits)

    for index in range(limit):
        if reference_is_relevant(hits[index].doc_id, relevant_doc_ids):
            return MetricScalar(1.0)

    return MetricScalar(0.0)


def reference_reciprocal_rank_at_k(
    hits: List[SearchHit], relevant_doc_ids: List[String], k: Int
) -> MetricScalar:
    var limit = k
    if limit > len(hits):
        limit = len(hits)

    for index in range(limit):
        if reference_is_relevant(hits[index].doc_id, relevant_doc_ids):
            return MetricScalar(1.0) / MetricScalar(index + 1)

    return MetricScalar(0.0)


def reference_recall_at_k(
    hits: List[SearchHit], relevant_doc_ids: List[String], k: Int
) -> MetricScalar:
    if len(relevant_doc_ids) == 0:
        return MetricScalar(0.0)

    var limit = k
    if limit > len(hits):
        limit = len(hits)

    var found_count = 0
    for index in range(limit):
        if reference_is_relevant(hits[index].doc_id, relevant_doc_ids):
            found_count += 1

    return MetricScalar(found_count) / MetricScalar(len(relevant_doc_ids))


def make_eval_fixture_task(primary_metric: String) raises -> JudgedTask:
    return JudgedTask(
        "mock",
        "eval_battle",
        "Metric battle fixture.",
        primary_metric,
        1,
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


def test_randomized_metrics_match_reference_formulas() raises:
    for seed in range(1, 31):
        var rng = Lcg(seed)
        var hit_count = 1 + rng.next_int(6)
        var hits = make_random_hits(rng, hit_count)
        var relevant_doc_ids = make_random_relevant_doc_ids(rng, hit_count)

        for k in range(hit_count + 2):
            assert_equal(
                success_at_k(hits, relevant_doc_ids, k),
                reference_success_at_k(hits, relevant_doc_ids, k),
            )
            assert_equal(
                reciprocal_rank_at_k(hits, relevant_doc_ids, k),
                reference_reciprocal_rank_at_k(hits, relevant_doc_ids, k),
            )
            assert_equal(
                recall_at_k(hits, relevant_doc_ids, k),
                reference_recall_at_k(hits, relevant_doc_ids, k),
            )


def test_success_and_recall_are_monotonic_in_k() raises:
    var hits = [
        SearchHit("doc-a", 4.0),
        SearchHit("doc-b", 3.0),
        SearchHit("doc-c", 2.0),
        SearchHit("doc-d", 1.0),
    ]
    var relevant_doc_ids = List[String]()
    relevant_doc_ids.append("doc-c")
    relevant_doc_ids.append("doc-d")

    var previous_success = MetricScalar(0.0)
    var previous_recall = MetricScalar(0.0)

    for k in range(len(hits) + 1):
        var current_success = success_at_k(hits, relevant_doc_ids, k)
        var current_recall = recall_at_k(hits, relevant_doc_ids, k)

        if current_success < previous_success:
            raise Error("success@k must be monotonic")
        if current_recall < previous_recall:
            raise Error("recall@k must be monotonic")

        previous_success = current_success
        previous_recall = current_recall


def test_reciprocal_rank_activates_at_first_relevant_hit() raises:
    var hits = [
        SearchHit("doc-a", 4.0),
        SearchHit("doc-b", 3.0),
        SearchHit("doc-c", 2.0),
    ]
    var relevant_doc_ids = List[String]()
    relevant_doc_ids.append("doc-b")

    assert_equal(reciprocal_rank_at_k(hits, relevant_doc_ids, 0), 0.0)
    assert_equal(reciprocal_rank_at_k(hits, relevant_doc_ids, 1), 0.0)
    assert_equal(reciprocal_rank_at_k(hits, relevant_doc_ids, 2), 0.5)
    assert_equal(reciprocal_rank_at_k(hits, relevant_doc_ids, 3), 0.5)


def test_evaluate_task_primary_metric_matches_selected_field() raises:
    var backend = ExactCpuBackend()

    var mrr_eval = evaluate_task(backend, make_eval_fixture_task("mrr"))
    assert_equal(mrr_eval.primary_value, mrr_eval.mean_reciprocal_rank)

    var recall_eval = evaluate_task(backend, make_eval_fixture_task("recall"))
    assert_equal(recall_eval.primary_value, recall_eval.mean_recall_at_k)

    var success_eval = evaluate_task(backend, make_eval_fixture_task("success"))
    assert_equal(success_eval.primary_value, success_eval.success_rate_at_k)


def test_evaluate_task_rejects_zero_queries() raises:
    var task = JudgedTask(
        "mock",
        "empty_queries",
        "Empty-query task should be rejected explicitly.",
        "mrr",
        2,
        1,
        1,
        2,
        [EncodedDocument("doc-a", [[1.0, 0.0]])],
        [],
    )

    var raised = False
    try:
        _ = evaluate_task(ExactCpuBackend(), task)
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
