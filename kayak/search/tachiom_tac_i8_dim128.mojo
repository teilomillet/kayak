from std.collections import List
from std.format import Writable, Writer
from std.sys.info import simd_width_of

from kayak.contracts import FlatQueryDim128
from kayak.index import HybridFlatDim128Index
from kayak.numeric import (
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at

from .hit import SearchHit
from .plaid_approx_dim128 import require_positive_int, top_positions_by_score
from .plaid_i8_approx_dim128 import (
    quantized_token_codes_dim128,
    quantized_token_scales_dim128,
)
from .topk import insert_descending, top_k_hits


# TAC candidate generation with an int8 token payload for candidate-window
# reranking. This is a compressed-rerank experiment, not the paper's PQ layout.
struct PreparedTachiomTacI8Index(Movable, Writable):
    var doc_ids: List[String]
    var doc_offsets: List[Int]
    var token_codes: List[Int8]
    var token_scales: List[ScoreScalar]
    var centroid_values: List[ScoreScalar]
    var centroid_doc_offsets: List[Int]
    var centroid_doc_indices: List[Int]
    var vector_dim: Int
    var document_count: Int
    var total_vector_count: Int
    var centroid_count: Int

    def __init__(
        out self,
        var doc_ids: List[String],
        var doc_offsets: List[Int],
        var token_codes: List[Int8],
        var token_scales: List[ScoreScalar],
        var centroid_values: List[ScoreScalar],
        var centroid_doc_offsets: List[Int],
        var centroid_doc_indices: List[Int],
        vector_dim: Int,
        document_count: Int,
        total_vector_count: Int,
    ) raises:
        if len(centroid_values) % COLBERT_VECTOR_DIM != 0:
            raise Error("Tachiom i8 centroid values must be aligned to dim128")
        var centroid_count = len(centroid_values) // COLBERT_VECTOR_DIM
        if len(centroid_doc_offsets) != centroid_count + 1:
            raise Error(
                "Tachiom i8 centroid_doc_offsets length must be centroid_count + 1"
            )
        if len(centroid_doc_offsets) == 0 or centroid_doc_offsets[0] != 0:
            raise Error("Tachiom i8 centroid_doc_offsets must start at zero")
        if centroid_doc_offsets[len(centroid_doc_offsets) - 1] != len(
            centroid_doc_indices
        ):
            raise Error("Tachiom i8 centroid_doc_offsets must end at posting count")

        self.doc_ids = doc_ids^
        self.doc_offsets = doc_offsets^
        self.token_codes = token_codes^
        self.token_scales = token_scales^
        self.centroid_values = centroid_values^
        self.centroid_doc_offsets = centroid_doc_offsets^
        self.centroid_doc_indices = centroid_doc_indices^
        self.vector_dim = vector_dim
        self.document_count = document_count
        self.total_vector_count = total_vector_count
        self.centroid_count = centroid_count

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedTachiomTacI8Index(document_count=",
            self.document_count,
            ", total_vector_count=",
            self.total_vector_count,
            ", centroid_count=",
            self.centroid_count,
            ")",
        )


def prepare_tachiom_tac_i8_hybrid_flat_dim128_index(
    read index: HybridFlatDim128Index,
    var centroid_values: List[ScoreScalar],
    var centroid_doc_offsets: List[Int],
    var centroid_doc_indices: List[Int],
) raises -> PreparedTachiomTacI8Index:
    var token_scales = quantized_token_scales_dim128(index)
    var token_codes = quantized_token_codes_dim128(index, token_scales)
    return PreparedTachiomTacI8Index(
        index.doc_ids.copy(),
        index.doc_offsets.copy(),
        token_codes^,
        token_scales^,
        centroid_values^,
        centroid_doc_offsets^,
        centroid_doc_indices^,
        index.vector_dim,
        index.document_count,
        index.total_vector_count,
    )


def tachiom_tac_i8_prepared_posting_count_value(
    read prepared_index: PreparedTachiomTacI8Index,
) -> Int:
    return len(prepared_index.centroid_doc_indices)


def score_query_vector_against_tachiom_i8_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacI8Index,
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


def tachiom_tac_i8_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    centroids_per_query_vector: Int,
    candidate_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)

    var document_scores = List[ScoreScalar]()
    var document_seen = List[Int]()
    for _ in range(prepared_index.document_count):
        document_scores.append(zero_score_scalar())
        document_seen.append(0)

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    for query_vector_index in range(query.vector_count):
        var centroid_scores = score_query_vector_against_tachiom_i8_centroids(
            query, query_vector_index, prepared_index
        )
        var centroid_positions = top_positions_by_score(
            centroid_scores, centroids_per_query_vector
        )
        var touched_documents = List[Int]()
        touched_documents.reserve(prepared_index.document_count)

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

    for document_index in range(prepared_index.document_count):
        if document_seen[document_index] == 0:
            document_scores[document_index] = min_score_scalar()

    return top_positions_by_score(document_scores, candidate_k)


def dot_query_vector_with_tachiom_i8_token_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacI8Index,
    token_index: Int,
) -> ScoreScalar:
    var query_offset = query_vector_index * COLBERT_VECTOR_DIM
    var token_offset = token_index * COLBERT_VECTOR_DIM

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        var total = Float64(0.0)
        for dim_index in range(COLBERT_VECTOR_DIM):
            total += Float64(
                query.token_values[query_offset + dim_index]
            ) * Float64(
                Int(prepared_index.token_codes[token_offset + dim_index])
            )

        return ScoreScalar(
            total * Float64(prepared_index.token_scales[token_index])
        )

    var accum = SIMD[DType.float32, width](0.0)
    var query_ptr = query.token_values.unsafe_ptr() + query_offset
    var code_ptr = prepared_index.token_codes.unsafe_ptr() + token_offset

    for dim_index in range(0, COLBERT_VECTOR_DIM, width):
        accum += (query_ptr + dim_index).load[width=width]() * (
            code_ptr + dim_index
        ).load[width=width]().cast[DType.float32]()

    return ScoreScalar(
        accum.reduce_add() * prepared_index.token_scales[token_index]
    )


def best_tachiom_i8_score_for_query_vector_in_document_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedTachiomTacI8Index,
    start_token: Int,
    stop_token: Int,
) -> ScoreScalar:
    var best_score = min_score_scalar()

    for token_index in range(start_token, stop_token):
        var score = dot_query_vector_with_tachiom_i8_token_dim128(
            query,
            query_vector_index,
            prepared_index,
            token_index,
        )
        if token_index == start_token or score > best_score:
            best_score = score

    return best_score


def tachiom_tac_i8_score_for_document(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    document_index: Int,
) -> ScoreScalar:
    var start_token = prepared_index.doc_offsets[document_index]
    var stop_token = prepared_index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_vector_index in range(query.vector_count):
        total += best_tachiom_i8_score_for_query_vector_in_document_dim128(
            query,
            query_vector_index,
            prepared_index,
            start_token,
            stop_token,
        )

    return total


def tachiom_tac_i8_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)

    var rerank_scores = List[ScoreScalar]()
    for document_index in candidate_positions:
        rerank_scores.append(
            tachiom_tac_i8_score_for_document(query, prepared_index, document_index)
        )

    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])

    return winners^


def tachiom_tac_i8_rerank_candidate_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[SearchHit]:
    require_positive_int("final_k", final_k)

    var hits = List[SearchHit]()
    for document_index in candidate_positions:
        insert_descending(
            hits,
            SearchHit(
                prepared_index.doc_ids[document_index].copy(),
                tachiom_tac_i8_score_for_document(
                    query, prepared_index, document_index
                ),
            ),
            final_k,
        )

    return hits^


def tachiom_tac_i8_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    final_k: Int,
) raises -> List[Int]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.document_count):
        scores.append(
            tachiom_tac_i8_score_for_document(query, prepared_index, document_index)
        )

    return top_positions_by_score(scores, final_k)


def tachiom_tac_i8_search_all_document_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    final_k: Int,
) raises -> List[SearchHit]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.document_count):
        scores.append(
            tachiom_tac_i8_score_for_document(query, prepared_index, document_index)
        )

    return top_k_hits(prepared_index.doc_ids, scores, final_k)


def tachiom_tac_i8_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.document_count:
        return tachiom_tac_i8_search_all_documents_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = tachiom_tac_i8_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return tachiom_tac_i8_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


def tachiom_tac_i8_search_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedTachiomTacI8Index,
    centroids_per_query_vector: Int,
    candidate_k: Int,
    final_k: Int,
) raises -> List[SearchHit]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)
    require_positive_int("final_k", final_k)

    if candidate_k >= prepared_index.document_count:
        return tachiom_tac_i8_search_all_document_hits_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = tachiom_tac_i8_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return tachiom_tac_i8_rerank_candidate_hits_for_query(
        query, prepared_index, candidate_positions, final_k
    )
