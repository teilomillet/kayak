from std.collections import List
from std.format import Writable, Writer
from std.os import abort
from std.pathlib import Path
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from kayak.contracts import EncodedQuery, FlatQueryDim128
from kayak.index import (
    HybridFlatDim128Index,
    PackedIndex,
    build_hybrid_flat_dim128_index,
)
from kayak.numeric import ScoreScalar, VectorScalar
from kayak.runtime import ExactCpuBackend
from kayak.search import (
    SearchHit,
    search_exact,
    search_exact_hybrid_flat_only_dim128,
)
from kayak.scoring import (
    ExactScoringConfig,
    exact_scores_for_hybrid_flat_index_dim128,
    exact_scores_for_hybrid_flat_index_dim128_with_flat_query,
    exact_scores_for_hybrid_flat_only_index_dim128,
)
from kayak.storage.binary_vector_codec import read_binary_scalar_payload
from kayak.storage.text_codec import read_non_empty_lines


struct PreparedPackedIndex(Movable, Writable):
    var index: HybridFlatDim128Index

    def __init__(out self, var index: HybridFlatDim128Index):
        self.index = index^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "PreparedPackedIndex(document_count=",
            self.index.document_count,
            ", vector_dim=",
            self.index.vector_dim,
            ")",
        )


def decode_string_list(py_values: PythonObject) raises -> List[String]:
    var values = List[String]()

    for index in range(len(py_values)):
        values.append(String(py=py_values[index]))

    return values^


def decode_int_list(py_values: PythonObject) raises -> List[Int]:
    var values = List[Int]()

    for index in range(len(py_values)):
        values.append(Int(py=py_values[index]))

    return values^


def decode_float_vector(py_values: PythonObject) raises -> List[VectorScalar]:
    var values = List[VectorScalar]()

    for index in range(len(py_values)):
        values.append(VectorScalar(py=py_values[index]))

    return values^


def decode_float_vectors(py_values: PythonObject) raises -> List[List[VectorScalar]]:
    var vectors = List[List[VectorScalar]]()

    for index in range(len(py_values)):
        vectors.append(decode_float_vector(py_values[index]))

    return vectors^


def decode_float_values(py_values: PythonObject) raises -> List[VectorScalar]:
    var values = List[VectorScalar]()

    for index in range(len(py_values)):
        values.append(VectorScalar(py=py_values[index]))

    return values^


def decode_query(py_query_vectors: PythonObject) raises -> EncodedQuery:
    return EncodedQuery(decode_float_vectors(py_query_vectors))


def decode_queries(py_query_batch_vectors: PythonObject) raises -> List[EncodedQuery]:
    var queries = List[EncodedQuery]()

    for index in range(len(py_query_batch_vectors)):
        queries.append(decode_query(py_query_batch_vectors[index]))

    return queries^


def decode_flat_query(py_query_values: PythonObject) raises -> FlatQueryDim128:
    return FlatQueryDim128(decode_float_values(py_query_values), 128)


def decode_packed_index(
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_vectors: PythonObject,
    vector_dim: Int,
) raises -> PackedIndex:
    return PackedIndex(
        decode_string_list(py_doc_ids),
        decode_int_list(py_doc_offsets),
        decode_float_vectors(py_token_vectors),
        vector_dim,
    )


def decode_hybrid_flat_dim128_index(
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_values: PythonObject,
) raises -> HybridFlatDim128Index:
    return HybridFlatDim128Index(
        decode_string_list(py_doc_ids),
        decode_int_list(py_doc_offsets),
        decode_float_values(py_token_values),
        128,
    )


def infer_packed_vector_dim(py_token_vectors: PythonObject) raises -> Int:
    if len(py_token_vectors) == 0:
        raise Error("packed index token_vectors must not be empty")
    return len(py_token_vectors[0])


def read_uint64_le(read bytes: List[Byte], offset: Int) raises -> UInt64:
    if offset + 8 > len(bytes):
        raise Error("binary int64 payload ended early")

    return (
        UInt64(bytes[offset])
        | (UInt64(bytes[offset + 1]) << 8)
        | (UInt64(bytes[offset + 2]) << 16)
        | (UInt64(bytes[offset + 3]) << 24)
        | (UInt64(bytes[offset + 4]) << 32)
        | (UInt64(bytes[offset + 5]) << 40)
        | (UInt64(bytes[offset + 6]) << 48)
        | (UInt64(bytes[offset + 7]) << 56)
    )


def read_binary_int64_payload(path: Path) raises -> List[Int]:
    var bytes = path.read_bytes()
    if len(bytes) % 8 != 0:
        raise Error("binary int64 payload byte length must be 8-byte aligned")

    var values = List[Int]()
    var cursor = 0
    while cursor < len(bytes):
        values.append(Int(read_uint64_le(bytes, cursor)))
        cursor += 8

    return values^


def scores_to_python(read scores: List[ScoreScalar]) raises -> PythonObject:
    var py_scores = Python.list()

    for score in scores:
        py_scores.append(Python.float(score))

    return py_scores


def scores_batch_to_python(
    read scores_by_query: List[List[ScoreScalar]]
) raises -> PythonObject:
    var py_scores_by_query = Python.list()

    for scores in scores_by_query:
        py_scores_by_query.append(scores_to_python(scores))

    return py_scores_by_query


def hits_to_python(read hits: List[SearchHit]) raises -> PythonObject:
    var py_hits = Python.list()

    for hit in hits:
        var py_hit = Python.list()
        py_hit.append(Python.str(hit.doc_id))
        py_hit.append(Python.float(hit.score))
        py_hits.append(py_hit)

    return py_hits


def hits_batch_to_python(
    read hits_by_query: List[List[SearchHit]]
) raises -> PythonObject:
    var py_hits_by_query = Python.list()

    for hits in hits_by_query:
        py_hits_by_query.append(hits_to_python(hits))

    return py_hits_by_query


def prepare_packed_index(
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_vectors: PythonObject,
) raises -> PythonObject:
    var vector_dim = infer_packed_vector_dim(py_token_vectors)
    var index = decode_packed_index(
        py_doc_ids,
        py_doc_offsets,
        py_token_vectors,
        vector_dim,
    )
    return PythonObject(
        alloc=PreparedPackedIndex(build_hybrid_flat_dim128_index(index))
    )


def prepare_packed_index_from_storage(
    py_root: PythonObject,
    py_vector_dim: PythonObject,
) raises -> PythonObject:
    var root = Path(String(py=py_root))
    var vector_dim = Int(py=py_vector_dim)
    if vector_dim != 128:
        raise Error("prepared packed storage path currently requires vector_dim=128")

    var index = HybridFlatDim128Index(
        read_non_empty_lines(root / "doc_ids.tsv"),
        read_binary_int64_payload(root / "doc_offsets.bin"),
        read_binary_scalar_payload(root / "token_values.bin"),
        vector_dim,
    )
    return PythonObject(alloc=PreparedPackedIndex(index^))


def exact_scores_packed(
    py_query_vectors: PythonObject,
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_vectors: PythonObject,
) raises -> PythonObject:
    var query = decode_query(py_query_vectors)
    var index = decode_packed_index(
        py_doc_ids, py_doc_offsets, py_token_vectors, query.vector_dim
    )

    var backend = ExactCpuBackend()
    return scores_to_python(backend.score_all(query, index))


def exact_scores_packed_batch(
    py_query_batch_vectors: PythonObject,
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_vectors: PythonObject,
) raises -> PythonObject:
    var queries = decode_queries(py_query_batch_vectors)
    if len(queries) == 0:
        return Python.list()

    var index = decode_packed_index(
        py_doc_ids, py_doc_offsets, py_token_vectors, queries[0].vector_dim
    )
    var backend = ExactCpuBackend()
    var scores_by_query = List[List[ScoreScalar]]()

    for query in queries:
        if query.vector_dim != index.vector_dim:
            raise Error("all queries must share the same vector dimension")

        scores_by_query.append(backend.score_all(query, index))

    return scores_batch_to_python(scores_by_query)


def exact_scores_prepared_packed(
    py_query_vectors: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var query = decode_query(py_query_vectors)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPackedIndex
    ]()
    if query.vector_dim != prepared_index[].index.vector_dim:
        raise Error("query vector dimension must match the prepared index")

    return scores_to_python(
        exact_scores_for_hybrid_flat_only_index_dim128(
            query, prepared_index[].index, ExactScoringConfig()
        )
    )


def exact_scores_prepared_packed_batch(
    py_query_batch_vectors: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_queries(py_query_batch_vectors)
    if len(queries) == 0:
        return Python.list()

    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPackedIndex
    ]()
    var scores_by_query = List[List[ScoreScalar]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        scores_by_query.append(
            exact_scores_for_hybrid_flat_only_index_dim128(
                query, prepared_index[].index, ExactScoringConfig()
            )
        )

    return scores_batch_to_python(scores_by_query)


def search_prepared_packed(
    py_query_vectors: PythonObject,
    py_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var query = decode_query(py_query_vectors)
    var k = Int(py=py_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPackedIndex
    ]()
    if query.vector_dim != prepared_index[].index.vector_dim:
        raise Error("query vector dimension must match the prepared index")

    return hits_to_python(
        search_exact_hybrid_flat_only_dim128(
            query, prepared_index[].index, k, ExactScoringConfig()
        )
    )


def search_prepared_packed_batch(
    py_query_batch_vectors: PythonObject,
    py_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_queries(py_query_batch_vectors)
    if len(queries) == 0:
        return Python.list()

    var k = Int(py=py_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPackedIndex
    ]()
    var hits_by_query = List[List[SearchHit]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        hits_by_query.append(
            search_exact_hybrid_flat_only_dim128(
                query, prepared_index[].index, k, ExactScoringConfig()
            )
        )

    return hits_batch_to_python(hits_by_query)


def exact_scores_hybrid_flat_dim128(
    py_query_vectors: PythonObject,
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_vectors: PythonObject,
    py_token_values: PythonObject,
) raises -> PythonObject:
    var query = decode_query(py_query_vectors)
    var nested_index = decode_packed_index(
        py_doc_ids, py_doc_offsets, py_token_vectors, query.vector_dim
    )
    var hybrid_index = decode_hybrid_flat_dim128_index(
        py_doc_ids, py_doc_offsets, py_token_values
    )

    return scores_to_python(
        exact_scores_for_hybrid_flat_index_dim128(
            query,
            nested_index,
            hybrid_index,
            ExactScoringConfig(),
        )
    )


def exact_scores_hybrid_flat_dim128_with_flat_query(
    py_query_values: PythonObject,
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_vectors: PythonObject,
    py_token_values: PythonObject,
) raises -> PythonObject:
    var query = decode_flat_query(py_query_values)
    var nested_index = decode_packed_index(
        py_doc_ids, py_doc_offsets, py_token_vectors, query.vector_dim
    )
    var hybrid_index = decode_hybrid_flat_dim128_index(
        py_doc_ids, py_doc_offsets, py_token_values
    )

    return scores_to_python(
        exact_scores_for_hybrid_flat_index_dim128_with_flat_query(
            query,
            nested_index,
            hybrid_index,
            ExactScoringConfig(),
        )
    )


@export
def PyInit__mojo_exact_cpu_bindings() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_exact_cpu_bindings")
        _ = module.add_type[PreparedPackedIndex]("PreparedPackedIndex")
        module.def_function[exact_scores_packed](
            "exact_scores_packed",
            docstring="Score a packed late-interaction index with the exact CPU backend.",
        )
        module.def_function[prepare_packed_index](
            "prepare_packed_index",
            docstring="Decode and retain a packed index inside a Mojo-backed Python object.",
        )
        module.def_function[prepare_packed_index_from_storage](
            "prepare_packed_index_from_storage",
            docstring="Load and retain a packed index from Kayak storage.",
        )
        module.def_function[exact_scores_packed_batch](
            "exact_scores_packed_batch",
            docstring="Score a packed late-interaction index for a whole batch of queries.",
        )
        module.def_function[exact_scores_prepared_packed](
            "exact_scores_prepared_packed",
            docstring="Score a prepared packed late-interaction index.",
        )
        module.def_function[exact_scores_prepared_packed_batch](
            "exact_scores_prepared_packed_batch",
            docstring="Score a prepared packed late-interaction index for a whole batch of queries.",
        )
        module.def_function[search_prepared_packed](
            "search_prepared_packed",
            docstring="Search a prepared packed late-interaction index.",
        )
        module.def_function[search_prepared_packed_batch](
            "search_prepared_packed_batch",
            docstring="Search a prepared packed late-interaction index for a whole batch of queries.",
        )
        module.def_function[exact_scores_hybrid_flat_dim128](
            "exact_scores_hybrid_flat_dim128",
            docstring="Score a hybrid flat dim128 index with a nested query.",
        )
        module.def_function[exact_scores_hybrid_flat_dim128_with_flat_query](
            "exact_scores_hybrid_flat_dim128_with_flat_query",
            docstring="Score a hybrid flat dim128 index with a flat dim128 query.",
        )
        return module.finalize()
    except e:
        abort(String("error creating kayak mojo exact cpu bindings:", e))
