from std.collections import List

from kayak.numeric import VectorScalar


def require_valid_packed_index(
    doc_ids: List[String],
    doc_offsets: List[Int],
    token_vectors: List[List[VectorScalar]],
    vector_dim: Int,
) raises:
    if len(doc_ids) == 0:
        raise Error("packed index must contain at least one document")

    if len(doc_offsets) != len(doc_ids) + 1:
        raise Error("packed index offsets must have document_count + 1 entries")

    if doc_offsets[0] != 0:
        raise Error("packed index offsets must start at zero")

    if vector_dim <= 0:
        raise Error("packed index vector_dim must be positive")

    if doc_offsets[len(doc_offsets) - 1] != len(token_vectors):
        raise Error("packed index offsets must end at total vector count")
    for vector in token_vectors:
        if len(vector) != vector_dim:
            raise Error("packed index token vector has the wrong dimension")
    for offset_index in range(1, len(doc_offsets)):
        if doc_offsets[offset_index] < doc_offsets[offset_index - 1]:
            raise Error("packed index offsets must be monotonic")
