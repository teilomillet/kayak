from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    ScoreScalar,
    VectorScalar,
    pack_documents,
    search_exact,
)
from kayak.numeric import min_score_scalar, zero_score_scalar
from kayak.search.topk import top_k_hits


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


def make_random_vector(mut rng: Lcg, vector_dim: Int) -> List[VectorScalar]:
    var vector = List[VectorScalar]()
    for _ in range(vector_dim):
        vector.append(VectorScalar(rng.next_int(4)))
    return vector^


def make_random_query(
    mut rng: Lcg, vector_dim: Int, vector_count: Int
) raises -> EncodedQuery:
    var token_vectors = List[List[VectorScalar]]()
    for _ in range(vector_count):
        token_vectors.append(make_random_vector(rng, vector_dim))
    return EncodedQuery(token_vectors^)


def make_random_documents(
    mut rng: Lcg, vector_dim: Int, document_count: Int
) raises -> List[EncodedDocument]:
    var documents = List[EncodedDocument]()

    for document_index in range(document_count):
        var token_vectors = List[List[VectorScalar]]()
        var vector_count = 1 + rng.next_int(4)

        for _ in range(vector_count):
            token_vectors.append(make_random_vector(rng, vector_dim))

        documents.append(
            EncodedDocument("doc-" + String(document_index), token_vectors^)
        )

    return documents^


def reference_dot(
    lhs: List[VectorScalar], rhs: List[VectorScalar]
) -> ScoreScalar:
    var total = zero_score_scalar()

    for index in range(len(lhs)):
        total += lhs[index] * rhs[index]

    return total


def reference_score(
    query: EncodedQuery, document: EncodedDocument
) -> ScoreScalar:
    var total = zero_score_scalar()

    for query_token in query.token_vectors:
        var best_similarity = min_score_scalar()

        for document_token in document.token_vectors:
            var similarity = reference_dot(query_token, document_token)
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def reference_scores(
    query: EncodedQuery, documents: List[EncodedDocument]
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for document in documents:
        scores.append(reference_score(query, document))
    return scores^


def reference_top_k(
    documents: List[EncodedDocument], scores: List[ScoreScalar], k: Int
) -> List[String]:
    var taken = List[Bool]()
    for _ in range(len(scores)):
        taken.append(False)

    var doc_ids = List[String]()
    var limit = k
    if limit > len(scores):
        limit = len(scores)

    for _ in range(limit):
        var found = False
        var best_index = 0
        var best_score = zero_score_scalar()

        for index in range(len(scores)):
            if taken[index]:
                continue

            if not found or scores[index] > best_score:
                found = True
                best_index = index
                best_score = scores[index]

        if not found:
            break

        taken[best_index] = True
        doc_ids.append(documents[best_index].doc_id.copy())

    return doc_ids^


def score_for_doc_id(
    doc_id: String, documents: List[EncodedDocument], scores: List[ScoreScalar]
) raises -> ScoreScalar:
    for index in range(len(documents)):
        if documents[index].doc_id == doc_id:
            return scores[index]

    raise Error("missing doc id in score lookup: " + doc_id)


def append_zero_vector_to_documents(
    documents: List[EncodedDocument]
) raises -> List[EncodedDocument]:
    var updated = List[EncodedDocument]()

    for document in documents:
        var token_vectors = List[List[VectorScalar]]()
        for token_vector in document.token_vectors:
            token_vectors.append(token_vector.copy())

        var zero_vector = List[VectorScalar]()
        for _ in range(document.vector_dim):
            zero_vector.append(VectorScalar(0.0))
        token_vectors.append(zero_vector^)

        updated.append(EncodedDocument(document.doc_id.copy(), token_vectors^))

    return updated^


def scale_query(query: EncodedQuery, factor: VectorScalar) raises -> EncodedQuery:
    var token_vectors = List[List[VectorScalar]]()

    for token_vector in query.token_vectors:
        var scaled = List[VectorScalar]()
        for value in token_vector:
            scaled.append(value * factor)
        token_vectors.append(scaled^)

    return EncodedQuery(token_vectors^)


def test_randomized_exact_search_matches_reference_scores() raises:
    var backend = ExactCpuBackend()

    for seed in range(1, 21):
        var rng = Lcg(seed)
        var vector_dim = 1 + rng.next_int(4)
        var query = make_random_query(rng, vector_dim, 1 + rng.next_int(4))
        var documents = make_random_documents(rng, vector_dim, 2 + rng.next_int(5))
        var index = pack_documents(documents)

        var actual_scores = backend.score_all(query, index)
        var expected_scores = reference_scores(query, documents)

        assert_equal(len(actual_scores), len(expected_scores))
        for index in range(len(actual_scores)):
            assert_equal(actual_scores[index], expected_scores[index])

        var actual_hits = search_exact(backend, query, index, 3)
        var expected_doc_ids = reference_top_k(documents, expected_scores, 3)

        assert_equal(len(actual_hits), len(expected_doc_ids))
        for index in range(len(actual_hits)):
            assert_equal(actual_hits[index].doc_id, expected_doc_ids[index])


def test_document_order_changes_positions_but_not_scores() raises:
    var backend = ExactCpuBackend()
    var query = EncodedQuery([[2.0, 1.0], [0.0, 3.0]])
    var documents = [
        EncodedDocument("doc-a", [[2.0, 0.0], [0.0, 3.0]]),
        EncodedDocument("doc-b", [[1.0, 1.0], [1.0, 0.0]]),
        EncodedDocument("doc-c", [[0.0, 3.0], [2.0, 1.0]]),
    ]
    var permuted = [
        documents[2].copy(),
        documents[0].copy(),
        documents[1].copy(),
    ]

    var original_scores = backend.score_all(
        query, pack_documents(documents)
    )
    var permuted_scores = backend.score_all(
        query, pack_documents(permuted)
    )

    for document in documents:
        assert_equal(
            score_for_doc_id(document.doc_id, documents, original_scores),
            score_for_doc_id(document.doc_id, permuted, permuted_scores),
        )


def test_appending_zero_document_vectors_preserves_nonnegative_scores() raises:
    var backend = ExactCpuBackend()
    var query = EncodedQuery([[1.0, 2.0], [0.0, 1.0]])
    var documents = [
        EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 2.0]]),
        EncodedDocument("doc-b", [[1.0, 1.0], [2.0, 0.0]]),
    ]
    var updated = append_zero_vector_to_documents(documents)

    var base_scores = backend.score_all(
        query, pack_documents(documents)
    )
    var updated_scores = backend.score_all(
        query, pack_documents(updated)
    )

    assert_equal(base_scores, updated_scores)


def test_positive_query_scaling_scales_scores_linearly() raises:
    var backend = ExactCpuBackend()
    var query = EncodedQuery([[1.0, 1.0], [2.0, 0.0]])
    var scaled_query = scale_query(query, VectorScalar(2.0))
    var documents = [
        EncodedDocument("doc-a", [[1.0, 0.0], [1.0, 1.0]]),
        EncodedDocument("doc-b", [[0.0, 2.0], [2.0, 0.0]]),
    ]

    var base_scores = backend.score_all(
        query, pack_documents(documents)
    )
    var scaled_scores = backend.score_all(
        scaled_query, pack_documents(documents)
    )

    for index in range(len(base_scores)):
        assert_equal(scaled_scores[index], base_scores[index] * 2.0)


def test_top_k_is_stable_for_equal_scores_and_handles_zero_k() raises:
    var hits = top_k_hits(
        ["doc-a", "doc-b", "doc-c"], [2.0, 2.0, 1.0], 2
    )
    assert_equal(len(hits), 2)
    assert_equal(hits[0].doc_id, "doc-a")
    assert_equal(hits[1].doc_id, "doc-b")

    var zero_hits = top_k_hits(["doc-a"], [1.0], 0)
    assert_equal(len(zero_hits), 0)


def test_top_k_rejects_negative_k_and_pack_documents_rejects_bad_inputs() raises:
    var negative_k_raised = False
    try:
        _ = top_k_hits(["doc-a"], [1.0], -1)
    except:
        negative_k_raised = True
    assert_equal(negative_k_raised, True)

    var empty_documents = List[EncodedDocument]()
    var empty_pack_raised = False
    try:
        _ = pack_documents(empty_documents)
    except:
        empty_pack_raised = True
    assert_equal(empty_pack_raised, True)

    var mismatched_documents = [
        EncodedDocument("doc-a", [[1.0, 0.0]]),
        EncodedDocument("doc-b", [[1.0, 0.0, 0.0]]),
    ]
    var mismatched_pack_raised = False
    try:
        _ = pack_documents(mismatched_documents)
    except:
        mismatched_pack_raised = True
    assert_equal(mismatched_pack_raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
