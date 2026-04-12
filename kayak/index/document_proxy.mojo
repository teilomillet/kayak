from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.numeric import VectorScalar, zero_vector_scalar

from .packed_index import PackedIndex


struct DocumentProxyIndex(Copyable):
    var doc_ids: List[String]
    var proxy_vectors: List[List[VectorScalar]]
    var vector_dim: Int
    var document_count: Int

    def __init__(
        out self,
        var doc_ids: List[String],
        var proxy_vectors: List[List[VectorScalar]],
        vector_dim: Int,
    ) raises:
        require_valid_document_proxy_index(doc_ids, proxy_vectors, vector_dim)

        self.doc_ids = doc_ids^
        self.proxy_vectors = proxy_vectors^
        self.vector_dim = vector_dim
        self.document_count = len(self.doc_ids)


def require_valid_document_proxy_index(
    read doc_ids: List[String],
    read proxy_vectors: List[List[VectorScalar]],
    vector_dim: Int,
) raises:
    if vector_dim <= 0:
        raise Error("document proxy index vector_dim must be positive")

    if len(doc_ids) != len(proxy_vectors):
        raise Error(
            "document proxy index doc_ids and proxy_vectors must have matching lengths"
        )

    for proxy_vector in proxy_vectors:
        if len(proxy_vector) != vector_dim:
            raise Error(
                "document proxy index proxy vector dimension does not match vector_dim"
            )


def effective_vector_budget(requested_budget: Int, available_count: Int) raises -> Int:
    if requested_budget < 0:
        raise Error("vector budget must be non-negative")

    if available_count <= 0:
        raise Error("vector budget requires at least one available vector")

    if requested_budget == 0 or requested_budget > available_count:
        return available_count

    return requested_budget


def average_vectors(
    read vectors: List[List[VectorScalar]], start: Int, stop: Int
) raises -> List[VectorScalar]:
    if start < 0 or stop <= start or stop > len(vectors):
        raise Error("invalid vector averaging range")

    var vector_dim = len(vectors[start])
    if vector_dim == 0:
        raise Error("proxy vectors must not be empty")

    var total = List[VectorScalar]()
    for _ in range(vector_dim):
        total.append(zero_vector_scalar())

    for vector_index in range(start, stop):
        if len(vectors[vector_index]) != vector_dim:
            raise Error("vector averaging requires a consistent dimension")

        for dim_index in range(vector_dim):
            total[dim_index] += vectors[vector_index][dim_index]

    var divisor = VectorScalar(stop - start)
    for dim_index in range(vector_dim):
        total[dim_index] = total[dim_index] / divisor

    return total^


def build_query_proxy_vector(
    read query: EncodedQuery, query_vector_budget: Int
) raises -> List[VectorScalar]:
    var budget = effective_vector_budget(query_vector_budget, query.vector_count)
    return average_vectors(query.token_vectors, 0, budget)


def build_document_proxy_index(
    read index: PackedIndex, document_vector_budget: Int
) raises -> DocumentProxyIndex:
    var proxy_vectors = List[List[VectorScalar]]()

    for document_index in range(index.document_count):
        var start = index.doc_offsets[document_index]
        var stop = index.doc_offsets[document_index + 1]
        var budget = effective_vector_budget(
            document_vector_budget, stop - start
        )
        proxy_vectors.append(average_vectors(index.token_vectors, start, start + budget))

    return DocumentProxyIndex(
        index.doc_ids.copy(),
        proxy_vectors^,
        index.vector_dim,
    )
