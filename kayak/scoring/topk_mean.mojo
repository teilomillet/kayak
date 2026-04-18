from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar

from .dot import dot_product


comptime LATE_INTERACTION_POOLING_KIND_MAXSIM = "maxsim"
comptime LATE_INTERACTION_POOLING_KIND_TOPK_MEAN = "topk_mean"


def require_late_interaction_pooling_kind_supported(kind: String) raises -> String:
    if kind == LATE_INTERACTION_POOLING_KIND_MAXSIM:
        return kind.copy()
    if kind == LATE_INTERACTION_POOLING_KIND_TOPK_MEAN:
        return kind.copy()

    raise Error("unsupported late interaction pooling kind: " + kind)


def insert_descending_score(
    mut values: List[ScoreScalar], score: ScoreScalar, k: Int
) raises:
    if k <= 0:
        raise Error("descending score insertion requires positive k")

    var insert_at = 0
    while insert_at < len(values) and values[insert_at] >= score:
        insert_at += 1

    if insert_at >= k:
        if len(values) < k:
            values.append(score)
        return

    if len(values) < k:
        values.append(score)

    var current = len(values) - 1
    while current > insert_at:
        values[current] = values[current - 1]
        current -= 1

    values[insert_at] = score


def topk_mean_query_token_score(
    query_token_index: Int,
    read query: EncodedQuery,
    read index: PackedIndex,
    start: Int,
    stop: Int,
    match_k: Int,
) raises -> ScoreScalar:
    if match_k <= 0:
        raise Error("top-k mean late interaction requires positive match_k")

    if start == stop:
        return min_score_scalar()

    var effective_k = match_k
    var document_token_count = stop - start
    if effective_k > document_token_count:
        effective_k = document_token_count

    var top_scores = List[ScoreScalar]()
    for token_index in range(start, stop):
        insert_descending_score(
            top_scores,
            dot_product(
                query.token_vectors[query_token_index],
                index.token_vectors[token_index],
            ),
            effective_k,
        )

    var total = zero_score_scalar()
    for score in top_scores:
        total += score
    return total / ScoreScalar(effective_k)


def topk_mean_score_for_document(
    read query: EncodedQuery,
    read index: PackedIndex,
    document_index: Int,
    match_k: Int,
) raises -> ScoreScalar:
    if query.vector_dim != index.vector_dim:
        raise Error("query and index must share the same vector dimension")
    if document_index < 0 or document_index >= index.document_count:
        raise Error("document_index is out of range")
    if match_k <= 0:
        raise Error("top-k mean late interaction requires positive match_k")

    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_token_index in range(query.vector_count):
        total += topk_mean_query_token_score(
            query_token_index,
            query,
            index,
            start,
            stop,
            match_k,
        )

    return total


def topk_mean_scores_for_index(
    read query: EncodedQuery,
    read index: PackedIndex,
    match_k: Int,
) raises -> List[ScoreScalar]:
    if query.vector_dim != index.vector_dim:
        raise Error("query and index must share the same vector dimension")
    if match_k <= 0:
        raise Error("top-k mean late interaction requires positive match_k")

    var scores = List[ScoreScalar]()
    for document_index in range(index.document_count):
        scores.append(
            topk_mean_score_for_document(
                query,
                index,
                document_index,
                match_k,
            )
        )
    return scores^
