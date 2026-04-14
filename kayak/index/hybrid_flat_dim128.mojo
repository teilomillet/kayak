from std.collections import List
from std.format import Writable, Writer

from kayak.numeric import VectorScalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM

from .packed_index import PackedIndex


# Owns the optional flat token-value layout for dim128 document indexes.
# It does not own query encoding or search orchestration.
struct HybridFlatDim128Index(Copyable, Writable):
    var doc_ids: List[String]
    var doc_offsets: List[Int]
    var token_values: List[VectorScalar]
    var vector_dim: Int
    var document_count: Int
    var total_vector_count: Int

    def __init__(
        out self,
        var doc_ids: List[String],
        var doc_offsets: List[Int],
        var token_values: List[VectorScalar],
        vector_dim: Int,
    ) raises:
        require_valid_hybrid_flat_dim128_index(
            doc_ids, doc_offsets, token_values, vector_dim
        )

        self.doc_ids = doc_ids^
        self.doc_offsets = doc_offsets^
        self.token_values = token_values^
        self.vector_dim = vector_dim
        self.document_count = len(self.doc_ids)
        self.total_vector_count = len(self.token_values) // COLBERT_VECTOR_DIM

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "HybridFlatDim128Index(document_count=",
            self.document_count,
            ", total_vector_count=",
            self.total_vector_count,
            ", vector_dim=",
            self.vector_dim,
            ")",
        )


def require_valid_hybrid_flat_dim128_index(
    read doc_ids: List[String],
    read doc_offsets: List[Int],
    read token_values: List[VectorScalar],
    vector_dim: Int,
) raises:
    if vector_dim != COLBERT_VECTOR_DIM:
        raise Error("hybrid flat dim128 index requires vector_dim=128")

    if len(doc_offsets) != len(doc_ids) + 1:
        raise Error("hybrid flat dim128 doc_offsets length must equal doc_ids + 1")

    if doc_offsets[0] != 0:
        raise Error("hybrid flat dim128 doc_offsets must start at 0")

    for offset_index in range(1, len(doc_offsets)):
        if doc_offsets[offset_index] < doc_offsets[offset_index - 1]:
            raise Error("hybrid flat dim128 doc_offsets must be monotonic")

    if len(token_values) % COLBERT_VECTOR_DIM != 0:
        raise Error("hybrid flat dim128 token_values length must be vector aligned")

    var total_vector_count = len(token_values) // COLBERT_VECTOR_DIM
    if doc_offsets[len(doc_offsets) - 1] != total_vector_count:
        raise Error("hybrid flat dim128 last doc offset must equal vector count")


def flatten_document_tokens(
    read document_tokens: List[List[VectorScalar]]
) -> List[VectorScalar]:
    var flat_values = List[VectorScalar]()

    for token in document_tokens:
        for value in token:
            flat_values.append(value)

    return flat_values^


def build_hybrid_flat_dim128_index(
    read index: PackedIndex
) raises -> HybridFlatDim128Index:
    return HybridFlatDim128Index(
        index.doc_ids.copy(),
        index.doc_offsets.copy(),
        flatten_document_tokens(index.token_vectors),
        index.vector_dim,
    )
