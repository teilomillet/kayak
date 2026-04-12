from std.collections import List
from std.sys.info import simd_width_of

from kayak.numeric import (
    VECTOR_SCALAR_NAME,
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)

from .dot128 import COLBERT_VECTOR_DIM


# Owns dim128 dot products against flat scalar buffers.
# It does not own document layout validation or search ranking.
def dot_product_dim128_flat_at(
    read lhs: List[VectorScalar],
    read flat_rhs: List[VectorScalar],
    rhs_offset: Int,
) -> ScoreScalar:
    if VECTOR_SCALAR_NAME != "Float32":
        var total = zero_score_scalar()

        for index in range(COLBERT_VECTOR_DIM):
            total += lhs[index] * flat_rhs[rhs_offset + index]

        return total

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        var total = zero_score_scalar()

        for index in range(COLBERT_VECTOR_DIM):
            total += lhs[index] * flat_rhs[rhs_offset + index]

        return total

    var accum = SIMD[DType.float32, width](0.0)
    var lhs_ptr = lhs.unsafe_ptr()
    var rhs_ptr = flat_rhs.unsafe_ptr() + rhs_offset

    for index in range(0, COLBERT_VECTOR_DIM, width):
        accum += (
            (lhs_ptr + index).load[width=width]()
            * (rhs_ptr + index).load[width=width]()
        )

    return ScoreScalar(accum.reduce_add()[0])


def dot_product_dim128_flat_pair_at(
    read flat_lhs: List[VectorScalar],
    lhs_offset: Int,
    read flat_rhs: List[VectorScalar],
    rhs_offset: Int,
) -> ScoreScalar:
    if VECTOR_SCALAR_NAME != "Float32":
        var total = zero_score_scalar()

        for index in range(COLBERT_VECTOR_DIM):
            total += flat_lhs[lhs_offset + index] * flat_rhs[rhs_offset + index]

        return total

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        var total = zero_score_scalar()

        for index in range(COLBERT_VECTOR_DIM):
            total += flat_lhs[lhs_offset + index] * flat_rhs[rhs_offset + index]

        return total

    var accum = SIMD[DType.float32, width](0.0)
    var lhs_ptr = flat_lhs.unsafe_ptr() + lhs_offset
    var rhs_ptr = flat_rhs.unsafe_ptr() + rhs_offset

    for index in range(0, COLBERT_VECTOR_DIM, width):
        accum += (
            (lhs_ptr + index).load[width=width]()
            * (rhs_ptr + index).load[width=width]()
        )

    return ScoreScalar(accum.reduce_add()[0])


def best_similarity_for_query_token_dim128_flat(
    read query_token: List[VectorScalar],
    read flat_document_tokens: List[VectorScalar],
    document_vector_count: Int,
) -> ScoreScalar:
    var best_similarity = min_score_scalar()

    for token_index in range(document_vector_count):
        var similarity = dot_product_dim128_flat_at(
            query_token,
            flat_document_tokens,
            token_index * COLBERT_VECTOR_DIM,
        )
        if similarity > best_similarity:
            best_similarity = similarity

    return best_similarity
