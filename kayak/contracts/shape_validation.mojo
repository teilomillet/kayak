from std.collections import List

from kayak.numeric import VectorScalar


def require_consistent_vector_dim(
    vectors: List[List[VectorScalar]], owner: String
) raises -> Int:
    if len(vectors) == 0:
        raise Error(owner + " must contain at least one vector")

    var vector_dim = len(vectors[0])
    if vector_dim == 0:
        raise Error(owner + " vectors must not be empty")

    for vector in vectors:
        if len(vector) != vector_dim:
            raise Error(owner + " has inconsistent vector dimensions")

    return vector_dim
