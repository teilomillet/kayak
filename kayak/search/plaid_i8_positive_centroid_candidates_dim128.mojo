from std.collections import List

from kayak.contracts import FlatQueryDim128
from kayak.numeric import (
    ScoreScalar,
    min_score_scalar,
    zero_score_scalar,
)

from .plaid_approx_dim128 import (
    require_positive_int,
    top_positions_by_score,
    top_unordered_positions_by_score,
)
from .plaid_i8_approx_dim128 import (
    PreparedPlaidApproxI8Index,
    score_query_vector_against_i8_centroids,
)


# Benchmark-only candidate generator that skips non-positive selected centroid
# postings. It does not own production search behavior; it exists to test
# whether negative proxy evidence adds useful candidate recall or just work.
def plaid_i8_positive_centroid_candidate_positions_for_query_unordered(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    centroids_per_query_vector: Int,
    candidate_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)

    if candidate_k >= prepared_index.document_count:
        var all_positions = List[Int]()
        all_positions.reserve(prepared_index.document_count)
        for document_index in range(prepared_index.document_count):
            all_positions.append(document_index)
        return all_positions^

    var document_scores = List[ScoreScalar]()
    document_scores.reserve(prepared_index.document_count)
    for _ in range(prepared_index.document_count):
        document_scores.append(zero_score_scalar())

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    token_best_scores.reserve(prepared_index.document_count)
    token_seen.reserve(prepared_index.document_count)
    for _ in range(prepared_index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    var touched_documents = List[Int]()
    touched_documents.reserve(prepared_index.document_count)

    for query_vector_index in range(query.vector_count):
        var centroid_scores = score_query_vector_against_i8_centroids(
            query, query_vector_index, prepared_index
        )
        var centroid_positions = top_positions_by_score(
            centroid_scores, centroids_per_query_vector
        )
        var touched_document_count = 0

        for centroid_position in centroid_positions:
            var centroid_score = centroid_scores[centroid_position]
            if centroid_score <= zero_score_scalar():
                continue

            var start_posting = prepared_index.centroid_doc_offsets[
                centroid_position
            ]
            var stop_posting = prepared_index.centroid_doc_offsets[
                centroid_position + 1
            ]
            for posting_index in range(start_posting, stop_posting):
                var document_index = prepared_index.centroid_doc_indices[
                    posting_index
                ]
                if token_seen[document_index] == 0:
                    token_best_scores[document_index] = centroid_score
                    token_seen[document_index] = 1
                    if touched_document_count == len(touched_documents):
                        touched_documents.append(document_index)
                    else:
                        touched_documents[
                            touched_document_count
                        ] = document_index
                    touched_document_count += 1
                elif centroid_score > token_best_scores[document_index]:
                    token_best_scores[document_index] = centroid_score

        for touched_offset in range(touched_document_count):
            var document_index = touched_documents[touched_offset]
            document_scores[document_index] += token_best_scores[document_index]
            token_best_scores[document_index] = min_score_scalar()
            token_seen[document_index] = 0

    return top_unordered_positions_by_score(document_scores, candidate_k)
