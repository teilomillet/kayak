from std.collections import List

from kayak.contracts import FlatQueryDim128
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar

from .plaid_approx_dim128 import top_positions_by_score
from .tachiom_tac_dim128 import prune_tachiom_candidate_positions_by_score
from .tachiom_tac_hnsw_pq_dim128 import (
    PreparedTachiomTacHnswPqIndex,
    hnsw_pq_query_centroid_score,
    tachiom_tac_hnsw_pq_centroid_positions_for_query_vector,
    tachiom_tac_hnsw_pq_score_for_document,
)
from .tachiom_tac_profile_types_dim128 import TachiomTacCandidateAccumulation


# Shared helpers for HNSW+PQ profiling. They are kept out of the production
# query module so benchmark-only accounting does not obscure the hot path.
def build_hnsw_pq_centroid_positions_by_query_vector(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    centroids_per_query_vector: Int,
    ef_search: Int,
) raises -> List[List[Int]]:
    var positions_by_query_vector = List[List[Int]]()
    positions_by_query_vector.reserve(query.vector_count)

    for query_vector_index in range(query.vector_count):
        positions_by_query_vector.append(
            tachiom_tac_hnsw_pq_centroid_positions_for_query_vector(
                query,
                query_vector_index,
                prepared_index,
                centroids_per_query_vector,
                ef_search,
            )
        )

    return positions_by_query_vector^


def accumulate_hnsw_pq_candidate_document_scores_from_centroids(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    read centroid_positions_by_query_vector: List[List[Int]],
) -> TachiomTacCandidateAccumulation:
    var document_scores = List[ScoreScalar]()
    var document_seen = List[Int]()
    for _ in range(prepared_index.pq.document_count):
        document_scores.append(zero_score_scalar())
        document_seen.append(0)

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.pq.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    var posting_visit_count = 0
    var selected_centroid_count = 0
    var touched_document_count = 0
    var touched_documents = List[Int]()
    touched_documents.reserve(prepared_index.pq.document_count)

    for query_vector_index in range(len(centroid_positions_by_query_vector)):
        selected_centroid_count += len(
            centroid_positions_by_query_vector[query_vector_index]
        )
        var current_touched_document_count = 0

        for offset in range(
            len(centroid_positions_by_query_vector[query_vector_index])
        ):
            var centroid_position = centroid_positions_by_query_vector[
                query_vector_index
            ][offset]
            var centroid_score = hnsw_pq_query_centroid_score(
                query, query_vector_index, prepared_index, centroid_position
            )
            var start_posting = prepared_index.pq.centroid_doc_offsets[
                centroid_position
            ]
            var stop_posting = prepared_index.pq.centroid_doc_offsets[
                centroid_position + 1
            ]
            posting_visit_count += stop_posting - start_posting

            for posting_index in range(start_posting, stop_posting):
                var document_index = prepared_index.pq.centroid_doc_indices[
                    posting_index
                ]
                document_seen[document_index] = 1
                if token_seen[document_index] == 0:
                    token_best_scores[document_index] = centroid_score
                    token_seen[document_index] = 1
                    if current_touched_document_count == len(touched_documents):
                        touched_documents.append(document_index)
                    else:
                        touched_documents[
                            current_touched_document_count
                        ] = document_index
                    current_touched_document_count += 1
                elif centroid_score > token_best_scores[document_index]:
                    token_best_scores[document_index] = centroid_score

        touched_document_count += current_touched_document_count
        for touched_offset in range(current_touched_document_count):
            var document_index = touched_documents[touched_offset]
            document_scores[document_index] += token_best_scores[document_index]
            token_best_scores[document_index] = min_score_scalar()
            token_seen[document_index] = 0

    var score_checksum = zero_score_scalar()
    var seen_document_count = 0
    for document_index in range(prepared_index.pq.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()
        else:
            seen_document_count += 1
            score_checksum += document_scores[document_index]

    return TachiomTacCandidateAccumulation(
        document_scores^,
        posting_visit_count,
        selected_centroid_count,
        touched_document_count,
        seen_document_count,
        score_checksum,
    )


def hnsw_pq_profile_candidate_positions_from_scores(
    read document_scores: List[ScoreScalar],
    read ranked_candidates: List[Int],
    final_k: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    if candidate_pruning_alpha <= zero_score_scalar():
        var kept = List[Int]()
        kept.reserve(len(ranked_candidates))
        for position in ranked_candidates:
            kept.append(position)
        return kept^

    return prune_tachiom_candidate_positions_by_score(
        ranked_candidates,
        document_scores,
        final_k,
        candidate_pruning_alpha,
    )


def hnsw_pq_rerank_scores_for_candidates(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacHnswPqIndex,
    read residual_score_table: List[ScoreScalar],
    read candidate_positions: List[Int],
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    scores.reserve(len(candidate_positions))
    for document_index in candidate_positions:
        scores.append(
            tachiom_tac_hnsw_pq_score_for_document(
                query, prepared_index, residual_score_table, document_index
            )
        )
    return scores^


def hnsw_pq_winners_from_rerank_scores(
    read candidate_positions: List[Int],
    read rerank_scores: List[ScoreScalar],
    final_k: Int,
) raises -> List[Int]:
    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    winners.reserve(len(winner_offsets))
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])
    return winners^


def regular_hnsw_pq_document_vector_count(
    read prepared_index: PreparedTachiomTacHnswPqIndex,
) -> Int:
    if prepared_index.pq.document_count <= 0:
        return 0

    var first_count = (
        prepared_index.pq.doc_offsets[1]
        - prepared_index.pq.doc_offsets[0]
    )
    for document_index in range(1, prepared_index.pq.document_count):
        var count = (
            prepared_index.pq.doc_offsets[document_index + 1]
            - prepared_index.pq.doc_offsets[document_index]
        )
        if count != first_count:
            return -1

    return first_count
