from std.collections import List
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from kayak.contracts import EncodedQuery, FlatQueryDim128
from kayak.index import HybridFlatDim128Index, PackedIndex
from kayak.numeric import ScoreScalar, VectorScalar
from kayak.runtime import ExactCpuBackend
from kayak.scoring import (
    ExactScoringConfig,
    exact_scores_for_hybrid_flat_index_dim128,
    exact_scores_for_hybrid_flat_index_dim128_with_flat_query,
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


def scores_to_python(read scores: List[ScoreScalar]) raises -> PythonObject:
    var py_scores = Python.list()

    for score in scores:
        py_scores.append(Python.float(score))

    return py_scores


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
        module.def_function[exact_scores_packed](
            "exact_scores_packed",
            docstring="Score a packed late-interaction index with the exact CPU backend.",
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
