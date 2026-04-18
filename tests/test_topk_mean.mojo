from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    exact_scores_for_index,
    pack_documents,
    topk_mean_score_for_document,
    topk_mean_scores_for_index,
)


def test_topk_mean_match_k_one_matches_maxsim_scores() raises:
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])
    var index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[1.0, 0.0], [1.0, 0.0]]),
        ]
    )

    var maxsim_scores = exact_scores_for_index(query.copy(), index.copy())
    var topk_scores = topk_mean_scores_for_index(query, index, 1)

    assert_equal(len(maxsim_scores), len(topk_scores))
    for index in range(len(maxsim_scores)):
        assert_equal(maxsim_scores[index], topk_scores[index])


def test_topk_mean_rewards_multiple_supporting_matches() raises:
    var query = EncodedQuery([[1.0, 0.0]])
    var index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 0.0]]),
            EncodedDocument("doc-b", [[1.0, 0.0], [0.0, 1.0]]),
        ]
    )

    var scores = topk_mean_scores_for_index(query, index, 2)

    assert_equal(scores[0] > scores[1], True)
    assert_equal(scores[0], 1.0)
    assert_equal(scores[1], 0.5)


def test_topk_mean_clamps_match_k_to_document_length() raises:
    var query = EncodedQuery([[1.0, 0.0]])
    var index = pack_documents(
        [
            EncodedDocument("doc-a", [[1.0, 0.0]]),
        ]
    )

    assert_equal(topk_mean_score_for_document(query, index, 0, 4), 1.0)


def test_topk_mean_rejects_non_positive_match_k() raises:
    var raised = False
    var query = EncodedQuery([[1.0, 0.0]])
    var index = pack_documents([EncodedDocument("doc-a", [[1.0, 0.0]])])

    try:
        _ = topk_mean_scores_for_index(query, index, 0)
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
