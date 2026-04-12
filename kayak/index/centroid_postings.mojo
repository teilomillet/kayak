from std.collections import List

from kayak.numeric import VectorScalar, zero_vector_scalar

from .packed_index import PackedIndex


struct CentroidPostingIndex(Copyable):
    var centroid_dims: List[Int]
    var centroid_vectors: List[List[VectorScalar]]
    var posting_offsets: List[Int]
    var posting_doc_indices: List[Int]
    var posting_weights: List[Int]
    var vector_dim: Int
    var document_count: Int
    var centroid_count: Int
    var total_posting_count: Int

    def __init__(
        out self,
        var centroid_dims: List[Int],
        var centroid_vectors: List[List[VectorScalar]],
        var posting_offsets: List[Int],
        var posting_doc_indices: List[Int],
        var posting_weights: List[Int],
        vector_dim: Int,
        document_count: Int,
    ) raises:
        require_valid_centroid_posting_index(
            centroid_dims,
            centroid_vectors,
            posting_offsets,
            posting_doc_indices,
            posting_weights,
            vector_dim,
            document_count,
        )

        self.centroid_dims = centroid_dims^
        self.centroid_vectors = centroid_vectors^
        self.posting_offsets = posting_offsets^
        self.posting_doc_indices = posting_doc_indices^
        self.posting_weights = posting_weights^
        self.vector_dim = vector_dim
        self.document_count = document_count
        self.centroid_count = len(self.centroid_dims)
        self.total_posting_count = len(self.posting_doc_indices)


def require_valid_centroid_posting_index(
    read centroid_dims: List[Int],
    read centroid_vectors: List[List[VectorScalar]],
    read posting_offsets: List[Int],
    read posting_doc_indices: List[Int],
    read posting_weights: List[Int],
    vector_dim: Int,
    document_count: Int,
) raises:
    if vector_dim <= 0:
        raise Error("centroid posting index vector_dim must be positive")

    if document_count < 0:
        raise Error("centroid posting index document_count must be non-negative")

    if len(centroid_dims) != len(centroid_vectors):
        raise Error(
            "centroid posting index centroid_dims and centroid_vectors must match"
        )

    if len(posting_offsets) != len(centroid_dims) + 1:
        raise Error(
            "centroid posting index posting_offsets length must equal centroids + 1"
        )

    if len(posting_doc_indices) != len(posting_weights):
        raise Error(
            "centroid posting index posting doc indices and weights must match"
        )

    if posting_offsets[0] != 0:
        raise Error("centroid posting index posting_offsets must start at 0")

    for offset_index in range(1, len(posting_offsets)):
        if posting_offsets[offset_index] < posting_offsets[offset_index - 1]:
            raise Error(
                "centroid posting index posting_offsets must be monotonic"
            )

    if posting_offsets[len(posting_offsets) - 1] != len(posting_doc_indices):
        raise Error(
            "centroid posting index last posting offset must equal posting count"
        )

    for centroid_dim in centroid_dims:
        if centroid_dim < 0 or centroid_dim >= vector_dim:
            raise Error("centroid posting index centroid_dim must fit vector_dim")

    for centroid_vector in centroid_vectors:
        if len(centroid_vector) != vector_dim:
            raise Error(
                "centroid posting index centroid vector must match vector_dim"
            )

    for posting_weight in posting_weights:
        if posting_weight <= 0:
            raise Error("centroid posting index posting weights must be positive")

    for doc_index in posting_doc_indices:
        if doc_index < 0 or doc_index >= document_count:
            raise Error(
                "centroid posting index posting doc index must fit document_count"
            )


def zero_vector(vector_dim: Int) -> List[VectorScalar]:
    var values = List[VectorScalar]()

    for _ in range(vector_dim):
        values.append(zero_vector_scalar())

    return values^


def divide_vector(
    read values: List[VectorScalar], divisor: Int
) raises -> List[VectorScalar]:
    if divisor <= 0:
        raise Error("centroid divisor must be positive")

    var divided = List[VectorScalar]()
    var scalar_divisor = VectorScalar(divisor)

    for value in values:
        divided.append(value / scalar_divisor)

    return divided^


def dominant_dimension(read vector: List[VectorScalar]) raises -> Int:
    if len(vector) == 0:
        raise Error("dominant dimension requires a non-empty vector")

    var best_index = 0
    var best_value = vector[0]

    for dim_index in range(1, len(vector)):
        if vector[dim_index] > best_value:
            best_value = vector[dim_index]
            best_index = dim_index

    return best_index


def effective_centroid_budget(
    requested_budget: Int, available_count: Int
) raises -> Int:
    if requested_budget < 0:
        raise Error("centroid budget must be non-negative")

    if available_count <= 0:
        raise Error("centroid budget requires at least one available centroid")

    if requested_budget == 0 or requested_budget > available_count:
        return available_count

    return requested_budget


def append_dim_descending_by_count(
    mut sorted_dims: List[Int],
    read token_counts_by_dim: List[Int],
    dim: Int,
):
    var insert_at = 0
    while (
        insert_at < len(sorted_dims)
        and token_counts_by_dim[sorted_dims[insert_at]] >= token_counts_by_dim[dim]
    ):
        insert_at += 1

    sorted_dims.append(dim)
    var current = len(sorted_dims) - 1
    while current > insert_at:
        sorted_dims[current] = sorted_dims[current - 1]
        current -= 1

    sorted_dims[insert_at] = dim


def selected_centroid_dims(
    read token_counts_by_dim: List[Int], centroid_budget: Int
) raises -> List[Int]:
    var non_empty_dims = List[Int]()

    for dim in range(len(token_counts_by_dim)):
        if token_counts_by_dim[dim] > 0:
            append_dim_descending_by_count(non_empty_dims, token_counts_by_dim, dim)

    if len(non_empty_dims) == 0:
        raise Error("centroid posting index requires at least one non-empty centroid")

    var budget = effective_centroid_budget(centroid_budget, len(non_empty_dims))
    var selected_dims = List[Int]()
    for dim_index in range(budget):
        selected_dims.append(non_empty_dims[dim_index])

    return selected_dims^


def build_centroid_posting_index(
    read index: PackedIndex, centroid_budget: Int
) raises -> CentroidPostingIndex:
    var token_counts_by_dim = List[Int]()
    var sum_vectors_by_dim = List[List[VectorScalar]]()
    var posting_doc_indices_by_dim = List[List[Int]]()
    var posting_weights_by_dim = List[List[Int]]()

    for _ in range(index.vector_dim):
        token_counts_by_dim.append(0)
        sum_vectors_by_dim.append(zero_vector(index.vector_dim))
        posting_doc_indices_by_dim.append(List[Int]())
        posting_weights_by_dim.append(List[Int]())

    for document_index in range(index.document_count):
        var start = index.doc_offsets[document_index]
        var stop = index.doc_offsets[document_index + 1]
        var token_counts_for_document = List[Int]()
        for _ in range(index.vector_dim):
            token_counts_for_document.append(0)

        for token_index in range(start, stop):
            var dim = dominant_dimension(index.token_vectors[token_index])
            token_counts_for_document[dim] += 1
            token_counts_by_dim[dim] += 1

            for value_index in range(index.vector_dim):
                sum_vectors_by_dim[dim][value_index] += (
                    index.token_vectors[token_index][value_index]
                )

        for dim in range(index.vector_dim):
            if token_counts_for_document[dim] > 0:
                posting_doc_indices_by_dim[dim].append(document_index)
                posting_weights_by_dim[dim].append(token_counts_for_document[dim])

    var centroid_dims = selected_centroid_dims(token_counts_by_dim, centroid_budget)
    var centroid_vectors = List[List[VectorScalar]]()
    var posting_offsets = [0]
    var posting_doc_indices = List[Int]()
    var posting_weights = List[Int]()

    for dim in centroid_dims:
        centroid_vectors.append(
            divide_vector(sum_vectors_by_dim[dim], token_counts_by_dim[dim])
        )

        for posting_index in range(len(posting_doc_indices_by_dim[dim])):
            posting_doc_indices.append(posting_doc_indices_by_dim[dim][posting_index])
            posting_weights.append(posting_weights_by_dim[dim][posting_index])

        posting_offsets.append(len(posting_doc_indices))

    return CentroidPostingIndex(
        centroid_dims^,
        centroid_vectors^,
        posting_offsets^,
        posting_doc_indices^,
        posting_weights^,
        index.vector_dim,
        index.document_count,
    )
