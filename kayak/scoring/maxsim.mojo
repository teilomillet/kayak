from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar

from .dot import dot_product


def exact_score_for_document(
    query: EncodedQuery,
    index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_token in query.token_vectors:
        var best_similarity = min_score_scalar()

        for token_index in range(start, stop):
            var similarity = dot_product(
                query_token, index.token_vectors[token_index]
            )
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def exact_scores_for_index(
    query: EncodedQuery, index: PackedIndex
) raises -> List[ScoreScalar]:
    if query.vector_dim != index.vector_dim:
        raise Error("query and index must share the same vector dimension")

    var scores = List[ScoreScalar]()

    for document_index in range(index.document_count):
        scores.append(exact_score_for_document(query, index, document_index))

    return scores^
