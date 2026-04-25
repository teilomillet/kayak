from std.collections import List
from std.format import Writable, Writer
from std.math import abs
from std.sys.info import simd_width_of

from kayak.contracts import FlatQueryDim128
from kayak.index import HybridFlatDim128Index
from kayak.numeric import ScoreScalar, VectorScalar, min_score_scalar, zero_score_scalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM

from .hit import SearchHit
from .plaid_approx_dim128 import (
    build_centroid_doc_presence,
    centroid_doc_indices_from_presence,
    centroid_doc_offsets_from_presence,
    require_positive_int,
    sampled_centroid_token_indices,
    top_positions_by_score,
)
from .topk import insert_descending, top_k_hits


# Owns the compressed dim128 score-proxy lane for PLAID-style search. It stores
# int8 document tokens plus one scale per token and intentionally does not own
# exact float token values after build. This makes its bytes claim measurable
# separately from the exact-rerank lane.
struct PreparedPlaidApproxI8Index(Movable, Writable):
    var doc_ids: List[String]
    var doc_offsets: List[Int]
    var token_codes: List[Int8]
    var token_scales: List[ScoreScalar]
    var centroid_token_indices: List[Int]
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
        var centroid_token_indices: List[Int],
        var centroid_doc_offsets: List[Int],
        var centroid_doc_indices: List[Int],
        vector_dim: Int,
        document_count: Int,
        total_vector_count: Int,
    ):
        self.doc_ids = doc_ids^
        self.doc_offsets = doc_offsets^
        self.token_codes = token_codes^
        self.token_scales = token_scales^
        self.centroid_token_indices = centroid_token_indices^
        self.centroid_doc_offsets = centroid_doc_offsets^
        self.centroid_doc_indices = centroid_doc_indices^
        self.vector_dim = vector_dim
        self.document_count = document_count
        self.total_vector_count = total_vector_count
        self.centroid_count = len(self.centroid_token_indices)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedPlaidApproxI8Index(document_count=",
            self.document_count,
            ", total_vector_count=",
            self.total_vector_count,
            ", centroid_count=",
            self.centroid_count,
            ")",
        )


def token_absmax_dim128(
    read index: HybridFlatDim128Index, token_index: Int
) -> Float64:
    var token_offset = token_index * COLBERT_VECTOR_DIM
    var max_value = Float64(0.0)

    for dim_index in range(COLBERT_VECTOR_DIM):
        var value = abs(Float64(index.token_values[token_offset + dim_index]))
        if value > max_value:
            max_value = value

    return max_value


def quantize_i8_value(value: VectorScalar, scale: ScoreScalar) -> Int8:
    var scaled = Float64(value) / Float64(scale)
    var rounded: Int
    if scaled >= Float64(0.0):
        rounded = Int(scaled + Float64(0.5))
    else:
        rounded = Int(scaled - Float64(0.5))

    if rounded > 127:
        rounded = 127
    elif rounded < -127:
        rounded = -127

    return Int8(rounded)


def quantized_token_scales_dim128(
    read index: HybridFlatDim128Index
) -> List[ScoreScalar]:
    var token_scales = List[ScoreScalar]()

    for token_index in range(index.total_vector_count):
        var max_value = token_absmax_dim128(index, token_index)
        if max_value == Float64(0.0):
            token_scales.append(ScoreScalar(1.0))
        else:
            token_scales.append(ScoreScalar(max_value / Float64(127.0)))

    return token_scales^


def quantized_token_codes_dim128(
    read index: HybridFlatDim128Index, read token_scales: List[ScoreScalar]
) -> List[Int8]:
    var token_codes = List[Int8]()

    for token_index in range(index.total_vector_count):
        var token_offset = token_index * COLBERT_VECTOR_DIM
        var scale = token_scales[token_index]
        for dim_index in range(COLBERT_VECTOR_DIM):
            token_codes.append(
                quantize_i8_value(index.token_values[token_offset + dim_index], scale)
            )

    return token_codes^


def prepare_plaid_approx_i8_hybrid_flat_dim128_index(
    var index: HybridFlatDim128Index,
    centroid_count: Int,
) raises -> PreparedPlaidApproxI8Index:
    var centroid_token_indices = sampled_centroid_token_indices(
        index.total_vector_count, centroid_count
    )
    var centroid_doc_presence = build_centroid_doc_presence(
        index, centroid_token_indices
    )
    var centroid_doc_offsets = centroid_doc_offsets_from_presence(
        centroid_doc_presence,
        len(centroid_token_indices),
        index.document_count,
    )
    var centroid_doc_indices = centroid_doc_indices_from_presence(
        centroid_doc_presence,
        len(centroid_token_indices),
        index.document_count,
    )
    var token_scales = quantized_token_scales_dim128(index)
    var token_codes = quantized_token_codes_dim128(index, token_scales)
    return PreparedPlaidApproxI8Index(
        index.doc_ids.copy(),
        index.doc_offsets.copy(),
        token_codes^,
        token_scales^,
        centroid_token_indices^,
        centroid_doc_offsets^,
        centroid_doc_indices^,
        index.vector_dim,
        index.document_count,
        index.total_vector_count,
    )


def plaid_approx_i8_prepared_posting_count_value(
    read prepared_index: PreparedPlaidApproxI8Index,
) -> Int:
    return len(prepared_index.centroid_doc_indices)


def dot_query_vector_with_i8_token_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedPlaidApproxI8Index,
    token_index: Int,
) -> ScoreScalar:
    var query_offset = query_vector_index * COLBERT_VECTOR_DIM
    var token_offset = token_index * COLBERT_VECTOR_DIM

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        var total = Float64(0.0)
        for dim_index in range(COLBERT_VECTOR_DIM):
            total += (
                Float64(query.token_values[query_offset + dim_index])
                * Float64(Int(prepared_index.token_codes[token_offset + dim_index]))
            )

        return ScoreScalar(
            total * Float64(prepared_index.token_scales[token_index])
        )

    var accum = SIMD[DType.float32, width](0.0)
    var query_ptr = query.token_values.unsafe_ptr() + query_offset
    var code_ptr = prepared_index.token_codes.unsafe_ptr() + token_offset

    for dim_index in range(0, COLBERT_VECTOR_DIM, width):
        accum += (
            (query_ptr + dim_index).load[width=width]()
            * (code_ptr + dim_index).load[width=width]().cast[DType.float32]()
        )

    return ScoreScalar(
        accum.reduce_add()[0] * prepared_index.token_scales[token_index]
    )


def best_i8_score_for_query_vector_in_document_dim128(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedPlaidApproxI8Index,
    start_token: Int,
    stop_token: Int,
) -> ScoreScalar:
    var query_offset = query_vector_index * COLBERT_VECTOR_DIM
    var best_score = min_score_scalar()

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        for token_index in range(start_token, stop_token):
            var score = dot_query_vector_with_i8_token_dim128(
                query, query_vector_index, prepared_index, token_index
            )
            if score > best_score:
                best_score = score
        return best_score

    var query_ptr = query.token_values.unsafe_ptr() + query_offset
    var code_base_ptr = prepared_index.token_codes.unsafe_ptr()
    var token_index = start_token

    # Score four consecutive document tokens per query-vector pass so the
    # query vector is loaded once per dim block instead of once per token.
    while token_index + 3 < stop_token:
        var token_offset0 = token_index * COLBERT_VECTOR_DIM
        var token_offset1 = token_offset0 + COLBERT_VECTOR_DIM
        var token_offset2 = token_offset1 + COLBERT_VECTOR_DIM
        var token_offset3 = token_offset2 + COLBERT_VECTOR_DIM
        var code_ptr0 = code_base_ptr + token_offset0
        var code_ptr1 = code_base_ptr + token_offset1
        var code_ptr2 = code_base_ptr + token_offset2
        var code_ptr3 = code_base_ptr + token_offset3
        var accum0 = SIMD[DType.float32, width](0.0)
        var accum1 = SIMD[DType.float32, width](0.0)
        var accum2 = SIMD[DType.float32, width](0.0)
        var accum3 = SIMD[DType.float32, width](0.0)

        for dim_index in range(0, COLBERT_VECTOR_DIM, width):
            var query_values = (query_ptr + dim_index).load[width=width]()
            accum0 += (
                query_values
                * (code_ptr0 + dim_index).load[width=width]().cast[DType.float32]()
            )
            accum1 += (
                query_values
                * (code_ptr1 + dim_index).load[width=width]().cast[DType.float32]()
            )
            accum2 += (
                query_values
                * (code_ptr2 + dim_index).load[width=width]().cast[DType.float32]()
            )
            accum3 += (
                query_values
                * (code_ptr3 + dim_index).load[width=width]().cast[DType.float32]()
            )

        var score0 = ScoreScalar(
            accum0.reduce_add()[0] * prepared_index.token_scales[token_index]
        )
        if score0 > best_score:
            best_score = score0
        var score1 = ScoreScalar(
            accum1.reduce_add()[0] * prepared_index.token_scales[token_index + 1]
        )
        if score1 > best_score:
            best_score = score1
        var score2 = ScoreScalar(
            accum2.reduce_add()[0] * prepared_index.token_scales[token_index + 2]
        )
        if score2 > best_score:
            best_score = score2
        var score3 = ScoreScalar(
            accum3.reduce_add()[0] * prepared_index.token_scales[token_index + 3]
        )
        if score3 > best_score:
            best_score = score3

        token_index += 4

    while token_index < stop_token:
        var score = dot_query_vector_with_i8_token_dim128(
            query, query_vector_index, prepared_index, token_index
        )
        if score > best_score:
            best_score = score
        token_index += 1

    return best_score


def score_query_vector_against_i8_centroids(
    read query: FlatQueryDim128,
    query_vector_index: Int,
    read prepared_index: PreparedPlaidApproxI8Index,
) -> List[ScoreScalar]:
    var centroid_scores = List[ScoreScalar]()

    for centroid_index in range(prepared_index.centroid_count):
        centroid_scores.append(
            dot_query_vector_with_i8_token_dim128(
                query,
                query_vector_index,
                prepared_index,
                prepared_index.centroid_token_indices[centroid_index],
            )
        )

    return centroid_scores^


def plaid_i8_candidate_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    centroids_per_query_vector: Int,
    candidate_k: Int,
) raises -> List[Int]:
    require_positive_int(
        "centroids_per_query_vector", centroids_per_query_vector
    )
    require_positive_int("candidate_k", candidate_k)

    var document_scores = List[ScoreScalar]()
    for _ in range(prepared_index.document_count):
        document_scores.append(zero_score_scalar())

    var token_best_scores = List[ScoreScalar]()
    var token_seen = List[Int]()
    for _ in range(prepared_index.document_count):
        token_best_scores.append(min_score_scalar())
        token_seen.append(0)

    for query_vector_index in range(query.vector_count):
        var centroid_scores = score_query_vector_against_i8_centroids(
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
                elif centroid_score > token_best_scores[document_index]:
                    token_best_scores[document_index] = centroid_score

        for document_index in touched_documents:
            document_scores[document_index] += token_best_scores[document_index]
            token_best_scores[document_index] = min_score_scalar()
            token_seen[document_index] = 0

    return top_positions_by_score(document_scores, candidate_k)


def plaid_i8_score_for_document(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    document_index: Int,
) -> ScoreScalar:
    var start_token = prepared_index.doc_offsets[document_index]
    var stop_token = prepared_index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_vector_index in range(query.vector_count):
        total += best_i8_score_for_query_vector_in_document_dim128(
            query,
            query_vector_index,
            prepared_index,
            start_token,
            stop_token,
        )

    return total


def plaid_i8_rerank_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    read candidate_positions: List[Int],
    final_k: Int,
) raises -> List[Int]:
    require_positive_int("final_k", final_k)

    var rerank_scores = List[ScoreScalar]()
    for document_index in candidate_positions:
        rerank_scores.append(
            plaid_i8_score_for_document(query, prepared_index, document_index)
        )

    var winner_offsets = top_positions_by_score(rerank_scores, final_k)
    var winners = List[Int]()
    for winner_offset in winner_offsets:
        winners.append(candidate_positions[winner_offset])

    return winners^


def plaid_i8_scores_for_candidates_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    read candidate_positions: List[Int],
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for document_index in candidate_positions:
        scores.append(
            plaid_i8_score_for_document(query, prepared_index, document_index)
        )

    return scores^


def plaid_i8_rerank_candidate_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
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
                plaid_i8_score_for_document(query, prepared_index, document_index),
            ),
            final_k,
        )

    return hits^


def plaid_i8_search_all_documents_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    final_k: Int,
) raises -> List[Int]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.document_count):
        scores.append(plaid_i8_score_for_document(query, prepared_index, document_index))

    return top_positions_by_score(scores, final_k)


def plaid_i8_search_all_document_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
    final_k: Int,
) raises -> List[SearchHit]:
    var scores = List[ScoreScalar]()
    for document_index in range(prepared_index.document_count):
        scores.append(plaid_i8_score_for_document(query, prepared_index, document_index))

    return top_k_hits(prepared_index.doc_ids, scores, final_k)


def plaid_i8_search_positions_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
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
        return plaid_i8_search_all_documents_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = plaid_i8_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return plaid_i8_rerank_candidates_for_query(
        query, prepared_index, candidate_positions, final_k
    )


def plaid_i8_search_hits_for_query(
    read query: FlatQueryDim128,
    read prepared_index: PreparedPlaidApproxI8Index,
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
        return plaid_i8_search_all_document_hits_for_query(
            query, prepared_index, final_k
        )

    var candidate_positions = plaid_i8_candidate_positions_for_query(
        query, prepared_index, centroids_per_query_vector, candidate_k
    )
    return plaid_i8_rerank_candidate_hits_for_query(
        query, prepared_index, candidate_positions, final_k
    )
