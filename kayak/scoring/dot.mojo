from std.collections import List
from std.sys.info import simd_width_of

from kayak.numeric import (
    VECTOR_SCALAR_NAME,
    ScoreScalar,
    VectorScalar,
    zero_score_scalar,
)


def dot_product(
    lhs: List[VectorScalar], rhs: List[VectorScalar]
) -> ScoreScalar:
    if VECTOR_SCALAR_NAME != "Float32":
        var total = zero_score_scalar()

        for index in range(len(lhs)):
            total += lhs[index] * rhs[index]

        return total

    comptime width = simd_width_of[VectorScalar]()
    # Use the target SIMD width for the configured scalar type and fall back
    # to a scalar tail for any remainder.
    var accum = SIMD[DType.float32, width](0.0)
    var simd_limit = (len(lhs) // width) * width
    var lhs_ptr = lhs.unsafe_ptr()
    var rhs_ptr = rhs.unsafe_ptr()

    for index in range(0, simd_limit, width):
        accum += (
            (lhs_ptr + index).load[width=width]()
            * (rhs_ptr + index).load[width=width]()
        )

    var total = ScoreScalar(accum.reduce_add()[0])

    for index in range(simd_limit, len(lhs)):
        total += lhs.unsafe_get(index) * rhs.unsafe_get(index)

    return total
