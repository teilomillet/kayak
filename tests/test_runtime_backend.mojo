from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    JudgedQuery,
    JudgedTask,
    PackedIndex,
    ScoreScalar,
    pack_documents,
)
from kayak.eval import evaluate_task
from kayak.runtime import ExactScoringBackend
from kayak.search import search_exact


struct MockExactBackend(Copyable, ExactScoringBackend):
    def __init__(out self):
        pass

    def score_all(
        self, read query: EncodedQuery, read index: PackedIndex
    ) raises -> List[ScoreScalar]:
        _ = query
        if len(index.doc_ids) != 3:
            raise Error("mock backend expects exactly 3 documents")

        return [0.1, 0.9, 0.4]


def make_runtime_backend_task() raises -> JudgedTask:
    return JudgedTask(
        "mock",
        "runtime_backend",
        "Runtime backend contract fixture.",
        "mrr",
        2,
        1,
        1,
        2,
        [
            EncodedDocument("doc-a", [[1.0, 0.0]]),
            EncodedDocument("doc-b", [[0.0, 1.0]]),
            EncodedDocument("doc-c", [[0.5, 0.5]]),
        ],
        [
            JudgedQuery(
                "q-1",
                "mock query",
                EncodedQuery([[1.0, 0.0]]),
                ["doc-b"],
            )
        ],
    )


def test_search_exact_accepts_trait_conforming_backend() raises:
    var task = make_runtime_backend_task()
    var hits = search_exact(
        MockExactBackend(),
        task.queries[0].query,
        pack_documents(task.documents),
        task.k,
    )

    assert_equal(len(hits), 2)
    assert_equal(hits[0].doc_id, "doc-b")
    assert_equal(hits[1].doc_id, "doc-c")


def test_evaluate_task_accepts_trait_conforming_backend() raises:
    var evaluation = evaluate_task(MockExactBackend(), make_runtime_backend_task())

    assert_equal(evaluation.primary_metric, "mrr")
    assert_equal(evaluation.primary_value, 1.0)
    assert_equal(evaluation.mean_ndcg_at_k, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
