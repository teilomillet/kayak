from std.collections import List

from kayak.numeric import VectorScalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM

from .encoded import EncodedQuery


# Owns the optional flat dim128 query layout for multi-vector late interaction.
# It does not own encoding or ranking policy.
struct FlatQueryDim128(Copyable):
    var token_values: List[VectorScalar]
    var vector_dim: Int
    var vector_count: Int

    def __init__(
        out self,
        var token_values: List[VectorScalar],
        vector_dim: Int,
    ) raises:
        require_valid_flat_query_dim128(token_values, vector_dim)

        self.token_values = token_values^
        self.vector_dim = vector_dim
        self.vector_count = len(self.token_values) // COLBERT_VECTOR_DIM


def require_valid_flat_query_dim128(
    read token_values: List[VectorScalar], vector_dim: Int
) raises:
    if vector_dim != COLBERT_VECTOR_DIM:
        raise Error("flat dim128 query requires vector_dim=128")

    if len(token_values) == 0:
        raise Error("flat dim128 query must contain at least one vector")

    if len(token_values) % COLBERT_VECTOR_DIM != 0:
        raise Error("flat dim128 query token_values length must be vector aligned")


def flatten_query_tokens(
    read token_vectors: List[List[VectorScalar]]
) -> List[VectorScalar]:
    var flat_values = List[VectorScalar]()

    for token_vector in token_vectors:
        for value in token_vector:
            flat_values.append(value)

    return flat_values^


def build_flat_query_dim128(read query: EncodedQuery) raises -> FlatQueryDim128:
    return FlatQueryDim128(flatten_query_tokens(query.token_vectors), query.vector_dim)
