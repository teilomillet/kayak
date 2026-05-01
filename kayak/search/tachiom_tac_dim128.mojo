from std.collections import List
from std.format import Writable, Writer

from kayak.contracts import FlatQueryDim128
from kayak.index import HybridFlatDim128Index
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar
from kayak.scoring import (
    exact_score_for_hybrid_flat_document_dim128_with_flat_query,
)
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .hit import SearchHit
from .plaid_approx_dim128 import require_positive_int, top_positions_by_score
from .topk import insert_descending, top_k_hits


# Owns the Mojo execution primitive for the Tachiom TAC first gate. It consumes
# token-aware centroids built by Python and executes centroid scoring, posting
# accumulation, and exact candidate-window rerank in Mojo.
struct PreparedTachiomTacIndex(Movable, Writable):
    var index: HybridFlatDim128Index
    var centroid_values: List[ScoreScalar]
    var centroid_doc_offsets: List[Int]
    var centroid_doc_indices: List[Int]
    var centroid_count: Int

    def __init__(
        out self,
        var index: HybridFlatDim128Index,
        var centroid_values: List[ScoreScalar],
        var centroid_doc_offsets: List[Int],
        var centroid_doc_indices: List[Int],
    ) raises:
        if len(centroid_values) % COLBERT_VECTOR_DIM != 0:
            raise Error("Tachiom centroid values must be aligned to dim128")
        var centroid_count = len(centroid_values) // COLBERT_VECTOR_DIM
        if len(centroid_doc_offsets) != centroid_count + 1:
            raise Error(
                "Tachiom centroid_doc_offsets length must be centroid_count + 1"
            )
        if len(centroid_doc_offsets) == 0 or centroid_doc_offsets[0] != 0:
            raise Error("Tachiom centroid_doc_offsets must start at zero")
        if centroid_doc_offsets[len(centroid_doc_offsets) - 1] != len(
            centroid_doc_indices
        ):
            raise Error("Tachiom centroid_doc_offsets must end at posting count")

        self.index = index^
        self.centroid_values = centroid_values^
        self.centroid_doc_offsets = centroid_doc_offsets^
        self.centroid_doc_indices = centroid_doc_indices^
        self.centroid_count = centroid_count

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedTachiomTacIndex(document_count=",
            self.index.document_count,
            ", total_vector_count=",
            self.index.total_vector_count,
            ", centroid_count=",
            self.centroid_count,
            ")",
        )


def prepare_tachiom_tac_hybrid_flat_dim128_index(
    var index: HybridFlatDim128Index,
    var centroid_values: List[ScoreScalar],
    var centroid_doc_offsets: List[Int],
    var centroid_doc_indices: List[Int],
) raises -> PreparedTachiomTacIndex:
    return PreparedTachiomTacIndex(
        index^,
        centroid_values^,
        centroid_doc_offsets^,
        centroid_doc_indices^,
    )


def tachiom_tac_prepared_posting_count_value(
    read prepared_index: PreparedTachiomTacIndex,
) -> Int:
    return len(prepared_index.centroid_doc_indices)


def score_query_vector_against_tachiom_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacIndex,
) -> List[ScoreScalar]:
    var centroid_scores = List[ScoreScalar]()
    centroid_scores.reserve(prepared_index.centroid_count)
    var query_offset = query_vector_index * COLBERT_VECTOR_DIM

    for centroid_index in range(prepared_index.centroid_count):
        centroid_scores.append(
            dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                prepared_index.centroid_values,
                centroid_index * COLBERT_VECTOR_DIM,
            )
        )

    return centroid_scores^


def tachiom_tac_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
) raises -> List[Int]:
    var document_scores = tachiom_tac_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector
    )
    return top_positions_by_score(document_scores, candidate_k)


def tachiom_tac_candidate_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    var document_scores = tachiom_tac_document_scores_for_query(
        query, prepared_index, centroids_per_query_vector
    )
    var ranked_positions = top_positions_by_score(document_scores, candidate_k)
    return prune_tachiom_candidate_positions_by_score(
        ranked_positions,
        document_scores,
        final_k,
        candidate_pruning_alpha,
    )


def tachiom_tac_document_scores_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    centroids_per_query_vector: Int,
) raises -> List[ScoreScalar]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )

    var document_scores = List[ScoreScalar]()
    var document_seen = List[Int]()
    for _ in range(prepared_index.index.document_count):
        document_scores.append(zero_score_scalar())
        document_seen.append(0)

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    for query_vector_index in range(query.vector_count):
        var centroid_scores = score_query_vector_against_tachiom_centroids(
            query, query_vector_index, prepared_index
        )
        var centroid_positions = top_positions_by_score(
            centroid_scores, centroids_per_query_vector
        )
        var touched_documents = List[Int]()
        touched_documents.reserve(prepared_index.index.document_count)

        for centroid_position in centroid_positions:
            var centroid_score = centroid_scores[centroid_position]
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
                    touched_documents.append(document_index)
                    document_seen[document_index] = 1
                elif centroid_score > token_best_scores[document_index]:
                    token_best_scores[document_index] = centroid_score

        for document_index in touched_documents:
            document_scores[document_index] += token_best_scores[document_index]
            token_best_scores[document_index] = min_score_scalar()
            token_seen[document_index] = 0

    for document_index in range(prepared_index.index.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()

    return document_scores^


def prune_tachiom_candidate_positions_by_score(
    read ranked_positions: List[Int],
    read document_scores: List[ScoreScalar],
    final_k: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)

    var kept = List[Int]()
    if len(ranked_positions) == 0:
        return kept^
    if candidate_pruning_alpha <= zero_score_scalar():
        for position in ranked_positions:
            kept.append(position)
        return kept^

    var required_count = final_k
    if required_count > len(ranked_positions):
        required_count = len(ranked_positions)
    var cutoff_position = ranked_positions[required_count - 1]
    var cutoff_score = document_scores[cutoff_position]
    var threshold = (ScoreScalar(1.0) - candidate_pruning_alpha) * cutoff_score

    for offset in range(len(ranked_positions)):
        var position = ranked_positions[offset]
        if offset >= required_count and document_scores[position] < threshold:
            break
        kept.append(position)

    return kept^


def tachiom_tac_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)

    var rerank_scores = List[ScoreScalar]()
    for document_index in candidate_positions:
        rerank_scores.append(
            exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                query, prepared_index.index, document_index
            )
        )

    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])

    return winners^


def tachiom_tac_rerank_candidate_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[SearchHit]:
    require_positive_int("final_k", final_k)

    var hits = List[SearchHit]()
    for document_index in candidate_positions:
        var score = exact_score_for_hybrid_flat_document_dim128_with_flat_query(
            query, prepared_index.index, document_index
        )
        insert_descending(
            hits,
            SearchHit(prepared_index.index.doc_ids[document_index].copy(), score),
            final_k,
        )

    return hits^


def tachiom_tac_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    final_k: Int,
) raises -> List[Int]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.index.document_count):
        scores.append(
            exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                query, prepared_index.index, document_index
            )
        )

    return top_positions_by_score(scores, final_k)


def tachiom_tac_search_all_document_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    final_k: Int,
) raises -> List[SearchHit]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.index.document_count):
        scores.append(
            exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                query, prepared_index.index, document_index
            )
        )

    return top_k_hits(prepared_index.index.doc_ids, scores, final_k)


def tachiom_tac_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.index.document_count:
        return tachiom_tac_search_all_documents_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = tachiom_tac_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return tachiom_tac_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


def tachiom_tac_search_positions_for_query_with_pruning(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
    candidate_pruning_alpha: ScoreScalar,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.index.document_count:
        return tachiom_tac_search_all_documents_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = tachiom_tac_candidate_positions_for_query_with_pruning(
        query,
        prepared_index,
        centroids_per_query_vector,
        candidate_k,
        final_k,
        candidate_pruning_alpha,
    )
    return tachiom_tac_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


def tachiom_tac_search_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacIndex,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
) raises -> List[SearchHit]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.index.document_count:
        return tachiom_tac_search_all_document_hits_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = tachiom_tac_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return tachiom_tac_rerank_candidate_hits_for_query(
        query, prepared_index, candidate_positions, final_k
    )
