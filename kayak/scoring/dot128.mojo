from std.collections import List
from std.sys.info import simd_width_of

from kayak.numeric import (
    VECTOR_SCALAR_NAME,
    ScoreScalar,
    VectorScalar,
    zero_score_scalar,
)

comptime COLBERT_VECTOR_DIM = 128


def dot_product_dim128(
    lhs: List[VectorScalar], rhs: List[VectorScalar]
) -> ScoreScalar:
    if VECTOR_SCALAR_NAME != "Float32":
        var total = zero_score_scalar()

        for index in range(COLBERT_VECTOR_DIM):
            total += lhs[index] * rhs[index]

        return total

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        var total = zero_score_scalar()

        for index in range(COLBERT_VECTOR_DIM):
            total += lhs[index] * rhs[index]

        return total

    var accum = SIMD[DType.float32, width](0.0)
    var lhs_ptr = lhs.unsafe_ptr()
    var rhs_ptr = rhs.unsafe_ptr()

    for index in range(0, COLBERT_VECTOR_DIM, width):
        accum += (
            (lhs_ptr + index).load[width=width]()
            * (rhs_ptr + index).load[width=width]()
        )

    return ScoreScalar(accum.reduce_add()[0])
