from std.collections import List

from kayak.numeric import ScoreScalar, VectorScalar, zero_score_scalar


def dot_product(
    lhs: List[VectorScalar], rhs: List[VectorScalar]
) -> ScoreScalar:
    var total = zero_score_scalar()

    for index in range(len(lhs)):
        total += lhs[index] * rhs[index]

    return total
