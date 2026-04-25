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
    PreparedPlaidApproxI8Index,
    PreparedPlaidApproxIndex,
    SearchHit,
    plaid_approx_i8_prepared_posting_count_value,
    plaid_approx_prepared_posting_count_value,
    plaid_i8_search_hits_for_query,
    plaid_i8_search_positions_for_query,
    plaid_i8_candidate_positions_for_query,
    plaid_i8_scores_for_candidates_for_query,
    plaid_search_hits_for_query,
    plaid_search_positions_for_query,
    prepare_plaid_approx_i8_hybrid_flat_dim128_index,
    prepare_plaid_approx_hybrid_flat_dim128_index,
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


def decode_flat_queries(py_query_batch_values: PythonObject) raises -> List[FlatQueryDim128]:
    var queries = List[FlatQueryDim128]()

    for index in range(len(py_query_batch_values)):
        queries.append(decode_flat_query(py_query_batch_values[index]))

    return queries^


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


def positions_to_python(read positions: List[Int]) raises -> PythonObject:
    var py_positions = Python.list()

    for position in positions:
        py_positions.append(Python.int(position))

    return py_positions


def positions_batch_to_python(
    read positions_by_query: List[List[Int]]
) raises -> PythonObject:
    var py_positions_by_query = Python.list()

    for positions in positions_by_query:
        py_positions_by_query.append(positions_to_python(positions))

    return py_positions_by_query


def int_values_to_python(read values: List[Int]) raises -> PythonObject:
    var py_values = Python.list()

    for value in values:
        py_values.append(Python.int(value))

    return py_values


def int8_values_to_python(read values: List[Int8]) raises -> PythonObject:
    var py_values = Python.list()

    for value in values:
        py_values.append(Python.int(Int(value)))

    return py_values


def scalar_values_to_python(read values: List[ScoreScalar]) raises -> PythonObject:
    var py_values = Python.list()

    for value in values:
        py_values.append(Python.float(value))

    return py_values


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

    var token_values = List[VectorScalar]()
    var token_value_parts_path = root / "token_values.parts.tsv"
    if token_value_parts_path.exists():
        for part_name in read_non_empty_lines(token_value_parts_path):
            for value in read_binary_scalar_payload(root / part_name):
                token_values.append(value)
    else:
        token_values = read_binary_scalar_payload(root / "token_values.bin")

    var index = HybridFlatDim128Index(
        read_non_empty_lines(root / "doc_ids.tsv"),
        read_binary_int64_payload(root / "doc_offsets.bin"),
        token_values^,
        vector_dim,
    )
    return PythonObject(alloc=PreparedPackedIndex(index^))


def prepare_plaid_approx_hybrid_flat_dim128(
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_values: PythonObject,
    py_centroid_count: PythonObject,
) raises -> PythonObject:
    var centroid_count = Int(py=py_centroid_count)
    var index = decode_hybrid_flat_dim128_index(
        py_doc_ids, py_doc_offsets, py_token_values
    )
    return PythonObject(
        alloc=prepare_plaid_approx_hybrid_flat_dim128_index(
            index^, centroid_count
        )
    )


def prepare_plaid_approx_i8_hybrid_flat_dim128(
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_values: PythonObject,
    py_centroid_count: PythonObject,
) raises -> PythonObject:
    var centroid_count = Int(py=py_centroid_count)
    var index = decode_hybrid_flat_dim128_index(
        py_doc_ids, py_doc_offsets, py_token_values
    )
    return PythonObject(
        alloc=prepare_plaid_approx_i8_hybrid_flat_dim128_index(
            index^, centroid_count
        )
    )


def plaid_approx_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxIndex
    ]()
    return Python.int(plaid_approx_prepared_posting_count_value(prepared_index[]))


def plaid_approx_i8_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return Python.int(plaid_approx_i8_prepared_posting_count_value(prepared_index[]))


def plaid_i8_prepared_doc_offsets(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return int_values_to_python(prepared_index[].doc_offsets)


def plaid_i8_prepared_token_codes(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return int8_values_to_python(prepared_index[].token_codes)


def plaid_i8_prepared_token_scales(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return scalar_values_to_python(prepared_index[].token_scales)


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


def search_plaid_approx_prepared_batch(
    py_query_batch_values: PythonObject,
    py_final_k: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_flat_queries(py_query_batch_values)
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_final_k)
    var centroids_per_query_vector = Int(py=py_centroids_per_query_vector)
    var candidate_k = Int(py=py_candidate_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        positions_by_query.append(
            plaid_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_plaid_approx_prepared_hits_batch(
    py_query_batch_values: PythonObject,
    py_final_k: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_flat_queries(py_query_batch_values)
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_final_k)
    var centroids_per_query_vector = Int(py=py_centroids_per_query_vector)
    var candidate_k = Int(py=py_candidate_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxIndex
    ]()
    var hits_by_query = List[List[SearchHit]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        hits_by_query.append(
            plaid_search_hits_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return hits_batch_to_python(hits_by_query)


def search_plaid_approx_i8_prepared_batch(
    py_query_batch_values: PythonObject,
    py_final_k: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_flat_queries(py_query_batch_values)
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_final_k)
    var centroids_per_query_vector = Int(py=py_centroids_per_query_vector)
    var candidate_k = Int(py=py_candidate_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        positions_by_query.append(
            plaid_i8_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_plaid_approx_i8_prepared_hits_batch(
    py_query_batch_values: PythonObject,
    py_final_k: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_flat_queries(py_query_batch_values)
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_final_k)
    var centroids_per_query_vector = Int(py=py_centroids_per_query_vector)
    var candidate_k = Int(py=py_candidate_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    var hits_by_query = List[List[SearchHit]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        hits_by_query.append(
            plaid_i8_search_hits_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return hits_batch_to_python(hits_by_query)


def plaid_i8_candidate_positions_prepared_batch(
    py_query_batch_values: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_flat_queries(py_query_batch_values)
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_centroids_per_query_vector)
    var candidate_k = Int(py=py_candidate_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        positions_by_query.append(
            plaid_i8_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def plaid_i8_candidate_scores_prepared_batch(
    py_query_batch_values: PythonObject,
    py_candidate_positions_batch: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var queries = decode_flat_queries(py_query_batch_values)
    if len(queries) != len(py_candidate_positions_batch):
        raise Error("query count must match candidate position row count")
    if len(queries) == 0:
        return Python.list()

    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    var scores_by_query = List[List[ScoreScalar]]()

    for query_index in range(len(queries)):
        var query = queries[query_index].copy()
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error("all queries must share the prepared index vector dimension")

        var candidate_positions = decode_int_list(
            py_candidate_positions_batch[query_index]
        )
        scores_by_query.append(
            plaid_i8_scores_for_candidates_for_query(
                query,
                prepared_index[],
                candidate_positions,
            )
        )

    return scores_batch_to_python(scores_by_query)


@export
def PyInit__mojo_exact_cpu_bindings() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_exact_cpu_bindings")
        _ = module.add_type[PreparedPackedIndex]("PreparedPackedIndex")
        _ = module.add_type[PreparedPlaidApproxIndex]("PreparedPlaidApproxIndex")
        _ = module.add_type[PreparedPlaidApproxI8Index](
            "PreparedPlaidApproxI8Index"
        )
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
        module.def_function[prepare_plaid_approx_hybrid_flat_dim128](
            "prepare_plaid_approx_hybrid_flat_dim128",
            docstring="Prepare a sampled-centroid PLAID-style approximation index in Mojo.",
        )
        module.def_function[prepare_plaid_approx_i8_hybrid_flat_dim128](
            "prepare_plaid_approx_i8_hybrid_flat_dim128",
            docstring="Prepare an int8 sampled-centroid PLAID-style approximation index in Mojo.",
        )
        module.def_function[plaid_approx_prepared_posting_count](
            "plaid_approx_prepared_posting_count",
            docstring="Return the prepared PLAID-style index posting count.",
        )
        module.def_function[plaid_approx_i8_prepared_posting_count](
            "plaid_approx_i8_prepared_posting_count",
            docstring="Return the prepared int8 PLAID-style index posting count.",
        )
        module.def_function[plaid_i8_prepared_doc_offsets](
            "plaid_i8_prepared_doc_offsets",
            docstring="Return prepared int8 PLAID document offsets.",
        )
        module.def_function[plaid_i8_prepared_token_codes](
            "plaid_i8_prepared_token_codes",
            docstring="Return prepared int8 PLAID token codes.",
        )
        module.def_function[plaid_i8_prepared_token_scales](
            "plaid_i8_prepared_token_scales",
            docstring="Return prepared int8 PLAID token scales.",
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
        module.def_function[search_plaid_approx_prepared_batch](
            "search_plaid_approx_prepared_batch",
            docstring="Search a prepared sampled-centroid PLAID-style approximation index.",
        )
        module.def_function[search_plaid_approx_prepared_hits_batch](
            "search_plaid_approx_prepared_hits_batch",
            docstring="Search a prepared sampled-centroid PLAID-style approximation index and return hits.",
        )
        module.def_function[search_plaid_approx_i8_prepared_batch](
            "search_plaid_approx_i8_prepared_batch",
            docstring="Search a prepared int8 sampled-centroid PLAID-style approximation index.",
        )
        module.def_function[search_plaid_approx_i8_prepared_hits_batch](
            "search_plaid_approx_i8_prepared_hits_batch",
            docstring="Search a prepared int8 sampled-centroid PLAID-style approximation index and return hits.",
        )
        module.def_function[plaid_i8_candidate_positions_prepared_batch](
            "plaid_i8_candidate_positions_prepared_batch",
            docstring="Generate int8 PLAID candidate document positions without reranking.",
        )
        module.def_function[plaid_i8_candidate_scores_prepared_batch](
            "plaid_i8_candidate_scores_prepared_batch",
            docstring="Score provided candidate positions with the prepared int8 reranker.",
        )
        return module.finalize()
    except e:
        abort(String("error creating kayak mojo exact cpu bindings:", e))
