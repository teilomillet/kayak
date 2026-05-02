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
    PreparedTachiomTacI8Index,
    PreparedTachiomTacHnswIndex,
    PreparedTachiomTacHnswPqAddressIndex,
    PreparedTachiomTacHnswPqIndex,
    PreparedTachiomTacIndex,
    PreparedTachiomTacPqIndex,
    TachiomTacCandidateGenerationProfile,
    TachiomTacHnswPqQueryProfile,
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
    prepare_tachiom_tac_i8_hybrid_flat_dim128_index,
    prepare_tachiom_tac_hnsw_hybrid_flat_dim128_index,
    prepare_tachiom_tac_hnsw_pq_dim128_address_index,
    prepare_tachiom_tac_hnsw_pq_dim128_index,
    prepare_tachiom_tac_hybrid_flat_dim128_index,
    prepare_tachiom_tac_pq_dim128_index,
    profile_tachiom_tac_hnsw_pq_for_query,
    profile_tachiom_tac_candidate_generation_for_query,
    search_exact,
    search_exact_hybrid_flat_only_dim128,
    tachiom_tac_candidate_positions_for_query,
    tachiom_tac_candidate_positions_for_query_with_pruning,
    tachiom_tac_i8_candidate_positions_for_query,
    tachiom_tac_hnsw_candidate_positions_for_query,
    tachiom_tac_hnsw_candidate_positions_for_query_with_pruning,
    tachiom_tac_hnsw_pq_address_candidate_positions_for_query,
    tachiom_tac_hnsw_pq_address_candidate_positions_for_query_with_pruning,
    tachiom_tac_hnsw_pq_address_prepared_graph_edge_count_value,
    tachiom_tac_hnsw_pq_address_prepared_posting_count_value,
    tachiom_tac_hnsw_pq_address_search_positions_for_query,
    tachiom_tac_hnsw_pq_address_search_positions_for_query_with_pruning,
    tachiom_tac_hnsw_pq_candidate_positions_for_query,
    tachiom_tac_hnsw_pq_candidate_positions_for_query_with_pruning,
    tachiom_tac_hnsw_pq_prepared_graph_edge_count_value,
    tachiom_tac_hnsw_pq_prepared_posting_count_value,
    tachiom_tac_hnsw_pq_search_positions_for_query,
    tachiom_tac_hnsw_pq_search_positions_for_query_with_pruning,
    tachiom_tac_hnsw_prepared_graph_edge_count_value,
    tachiom_tac_hnsw_prepared_posting_count_value,
    tachiom_tac_hnsw_search_positions_for_query,
    tachiom_tac_i8_prepared_posting_count_value,
    tachiom_tac_i8_search_positions_for_query,
    tachiom_tac_pq_candidate_positions_for_query,
    tachiom_tac_pq_candidate_positions_for_query_with_pruning,
    tachiom_tac_pq_prepared_posting_count_value,
    tachiom_tac_pq_search_positions_for_query,
    tachiom_tac_pq_search_positions_for_query_with_pruning,
    tachiom_tac_prepared_posting_count_value,
    tachiom_tac_search_hits_for_query,
    tachiom_tac_search_positions_for_query,
    tachiom_tac_search_positions_for_query_with_pruning,
)
from kayak.search.plaid_i8_approx_dim128 import (
    plaid_i8_candidate_positions_for_query_unordered,
)
from kayak.search.plaid_i8_candidate_profile_dim128 import (
    build_i8_centroid_positions_by_query_vector,
    build_i8_centroid_scores_by_query_vector,
    profile_plaid_i8_candidate_generation_for_query,
)
from kayak.search.plaid_i8_candidate_profile_types_dim128 import (
    PlaidI8CandidateGenerationProfile,
)
from kayak.search.plaid_i8_positive_centroid_candidates_dim128 import (
    plaid_i8_positive_centroid_candidate_positions_for_query_unordered,
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


def decode_float_vectors(
    py_values: PythonObject,
) raises -> List[List[VectorScalar]]:
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


def decode_queries(
    py_query_batch_vectors: PythonObject,
) raises -> List[EncodedQuery]:
    var queries = List[EncodedQuery]()

    for index in range(len(py_query_batch_vectors)):
        queries.append(decode_query(py_query_batch_vectors[index]))

    return queries^


def decode_flat_query(py_query_values: PythonObject) raises -> FlatQueryDim128:
    return FlatQueryDim128(decode_float_values(py_query_values), 128)


def decode_flat_queries(
    py_query_batch_values: PythonObject,
) raises -> List[FlatQueryDim128]:
    var queries = List[FlatQueryDim128]()

    for index in range(len(py_query_batch_values)):
        queries.append(decode_flat_query(py_query_batch_values[index]))

    return queries^


def decode_flat_queries_from_float32_address(
    query_values_address: Int,
    query_count: Int,
    query_vector_count: Int,
) raises -> List[FlatQueryDim128]:
    if query_values_address == 0:
        raise Error("query_values address must be non-zero")
    if query_count < 0:
        raise Error("query_count must not be negative")
    if query_vector_count <= 0:
        raise Error("query_vector_count must be positive")

    var query_values = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=query_values_address
    )
    var query_value_count = query_vector_count * 128
    var queries = List[FlatQueryDim128]()
    queries.reserve(query_count)

    for query_index in range(query_count):
        var values = List[VectorScalar]()
        values.reserve(query_value_count)
        var query_base = query_index * query_value_count
        for offset in range(query_value_count):
            values.append(VectorScalar(query_values[query_base + offset]))
        queries.append(FlatQueryDim128(values^, 128))

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
    read scores_by_query: List[List[ScoreScalar]],
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
    read hits_by_query: List[List[SearchHit]],
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
    read positions_by_query: List[List[Int]],
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


def scalar_values_to_python(
    read values: List[ScoreScalar],
) raises -> PythonObject:
    var py_values = Python.list()

    for value in values:
        py_values.append(Python.float(value))

    return py_values


def append_profile_float(
    py_result: PythonObject, name: String, value: Float64
) raises:
    var field = Python.list()
    field.append(Python.str(name))
    field.append(Python.float(value))
    py_result.append(field)


def append_profile_int(
    py_result: PythonObject, name: String, value: Int
) raises:
    var field = Python.list()
    field.append(Python.str(name))
    field.append(Python.int(value))
    py_result.append(field)


def i8_candidate_generation_profile_to_python(
    read profile: PlaidI8CandidateGenerationProfile,
) raises -> PythonObject:
    var py_result = Python.list()
    append_profile_float(
        py_result,
        "full_candidate_mean_seconds",
        profile.full_candidate_mean_seconds,
    )
    append_profile_float(
        py_result,
        "workspace_full_candidate_mean_seconds",
        profile.workspace_full_candidate_mean_seconds,
    )
    append_profile_float(
        py_result,
        "workspace_candidate_position_agreement",
        profile.workspace_candidate_position_agreement,
    )
    append_profile_float(
        py_result,
        "unordered_candidate_mean_seconds",
        profile.unordered_candidate_mean_seconds,
    )
    append_profile_float(
        py_result,
        "unordered_candidate_set_agreement",
        profile.unordered_candidate_set_agreement,
    )
    append_profile_float(
        py_result,
        "centroid_scoring_mean_seconds",
        profile.centroid_scoring_mean_seconds,
    )
    append_profile_float(
        py_result,
        "centroid_selection_mean_seconds",
        profile.centroid_selection_mean_seconds,
    )
    append_profile_float(
        py_result,
        "unordered_centroid_selection_mean_seconds",
        profile.unordered_centroid_selection_mean_seconds,
    )
    append_profile_float(
        py_result,
        "centroid_selection_set_agreement",
        profile.centroid_selection_set_agreement,
    )
    append_profile_float(
        py_result,
        "posting_accumulation_mean_seconds",
        profile.posting_accumulation_mean_seconds,
    )
    append_profile_float(
        py_result,
        "final_topk_mean_seconds",
        profile.final_topk_mean_seconds,
    )
    append_profile_float(
        py_result,
        "unordered_final_topk_mean_seconds",
        profile.unordered_final_topk_mean_seconds,
    )
    append_profile_int(
        py_result, "query_vector_count", profile.query_vector_count
    )
    append_profile_int(py_result, "document_count", profile.document_count)
    append_profile_int(
        py_result, "document_vector_count", profile.document_vector_count
    )
    append_profile_int(
        py_result,
        "total_document_vector_count",
        profile.total_document_vector_count,
    )
    append_profile_int(py_result, "centroid_count", profile.centroid_count)
    append_profile_int(
        py_result,
        "centroids_per_query_vector",
        profile.centroids_per_query_vector,
    )
    append_profile_int(py_result, "candidate_k", profile.candidate_k)
    append_profile_int(
        py_result,
        "selected_centroid_count",
        profile.selected_centroid_count,
    )
    append_profile_int(
        py_result, "posting_visit_count", profile.posting_visit_count
    )
    append_profile_int(
        py_result,
        "touched_document_count",
        profile.touched_document_count,
    )
    append_profile_int(
        py_result, "output_candidate_count", profile.output_candidate_count
    )
    append_profile_int(
        py_result,
        "measurement_iterations",
        profile.measurement_iterations,
    )
    append_profile_float(py_result, "sink_value", profile.sink_value)
    return py_result


def tachiom_tac_candidate_generation_profile_to_python(
    read profile: TachiomTacCandidateGenerationProfile,
) raises -> PythonObject:
    var py_result = Python.list()
    append_profile_float(
        py_result,
        "full_candidate_mean_seconds",
        profile.full_candidate_mean_seconds,
    )
    append_profile_float(
        py_result,
        "full_search_mean_seconds",
        profile.full_search_mean_seconds,
    )
    append_profile_float(
        py_result,
        "centroid_scoring_mean_seconds",
        profile.centroid_scoring_mean_seconds,
    )
    append_profile_float(
        py_result,
        "centroid_selection_mean_seconds",
        profile.centroid_selection_mean_seconds,
    )
    append_profile_float(
        py_result,
        "posting_accumulation_mean_seconds",
        profile.posting_accumulation_mean_seconds,
    )
    append_profile_float(
        py_result,
        "final_topk_mean_seconds",
        profile.final_topk_mean_seconds,
    )
    append_profile_float(
        py_result,
        "exact_rerank_mean_seconds",
        profile.exact_rerank_mean_seconds,
    )
    append_profile_int(
        py_result, "query_vector_count", profile.query_vector_count
    )
    append_profile_int(py_result, "document_count", profile.document_count)
    append_profile_int(
        py_result, "document_vector_count", profile.document_vector_count
    )
    append_profile_int(
        py_result,
        "total_document_vector_count",
        profile.total_document_vector_count,
    )
    append_profile_int(py_result, "centroid_count", profile.centroid_count)
    append_profile_int(
        py_result,
        "centroids_per_query_vector",
        profile.centroids_per_query_vector,
    )
    append_profile_int(py_result, "candidate_k", profile.candidate_k)
    append_profile_int(py_result, "final_k", profile.final_k)
    append_profile_int(
        py_result,
        "selected_centroid_count",
        profile.selected_centroid_count,
    )
    append_profile_int(
        py_result, "posting_visit_count", profile.posting_visit_count
    )
    append_profile_int(
        py_result,
        "touched_document_count",
        profile.touched_document_count,
    )
    append_profile_int(
        py_result,
        "seen_document_count",
        profile.seen_document_count,
    )
    append_profile_int(
        py_result, "output_candidate_count", profile.output_candidate_count
    )
    append_profile_int(
        py_result, "output_final_count", profile.output_final_count
    )
    append_profile_int(
        py_result,
        "measurement_iterations",
        profile.measurement_iterations,
    )
    append_profile_float(py_result, "sink_value", profile.sink_value)
    return py_result


def tachiom_tac_hnsw_pq_query_profile_to_python(
    read profile: TachiomTacHnswPqQueryProfile,
) raises -> PythonObject:
    var py_result = Python.list()
    append_profile_float(
        py_result,
        "full_search_mean_seconds",
        profile.full_search_mean_seconds,
    )
    append_profile_float(
        py_result,
        "candidate_generation_mean_seconds",
        profile.candidate_generation_mean_seconds,
    )
    append_profile_float(
        py_result,
        "hnsw_traversal_mean_seconds",
        profile.hnsw_traversal_mean_seconds,
    )
    append_profile_float(
        py_result,
        "candidate_score_accumulation_mean_seconds",
        profile.candidate_score_accumulation_mean_seconds,
    )
    append_profile_float(
        py_result,
        "candidate_topk_mean_seconds",
        profile.candidate_topk_mean_seconds,
    )
    append_profile_float(
        py_result,
        "candidate_pruning_mean_seconds",
        profile.candidate_pruning_mean_seconds,
    )
    append_profile_float(
        py_result,
        "residual_score_table_mean_seconds",
        profile.residual_score_table_mean_seconds,
    )
    append_profile_float(
        py_result,
        "rerank_scoring_mean_seconds",
        profile.rerank_scoring_mean_seconds,
    )
    append_profile_float(
        py_result,
        "rerank_topk_mean_seconds",
        profile.rerank_topk_mean_seconds,
    )
    append_profile_int(
        py_result, "query_vector_count", profile.query_vector_count
    )
    append_profile_int(py_result, "document_count", profile.document_count)
    append_profile_int(
        py_result, "document_vector_count", profile.document_vector_count
    )
    append_profile_int(
        py_result,
        "total_document_vector_count",
        profile.total_document_vector_count,
    )
    append_profile_int(py_result, "centroid_count", profile.centroid_count)
    append_profile_int(
        py_result,
        "centroids_per_query_vector",
        profile.centroids_per_query_vector,
    )
    append_profile_int(py_result, "candidate_k", profile.candidate_k)
    append_profile_int(py_result, "final_k", profile.final_k)
    append_profile_int(py_result, "ef_search", profile.ef_search)
    append_profile_float(
        py_result,
        "candidate_pruning_alpha",
        profile.candidate_pruning_alpha,
    )
    append_profile_int(
        py_result,
        "selected_centroid_count",
        profile.selected_centroid_count,
    )
    append_profile_int(
        py_result, "posting_visit_count", profile.posting_visit_count
    )
    append_profile_int(
        py_result,
        "touched_document_count",
        profile.touched_document_count,
    )
    append_profile_int(
        py_result,
        "seen_document_count",
        profile.seen_document_count,
    )
    append_profile_int(
        py_result,
        "ranked_candidate_count",
        profile.ranked_candidate_count,
    )
    append_profile_int(
        py_result, "output_candidate_count", profile.output_candidate_count
    )
    append_profile_int(
        py_result, "output_final_count", profile.output_final_count
    )
    append_profile_int(
        py_result,
        "measurement_iterations",
        profile.measurement_iterations,
    )
    append_profile_float(py_result, "sink_value", profile.sink_value)
    return py_result


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
        raise Error(
            "prepared packed storage path currently requires vector_dim=128"
        )

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


def prepare_tachiom_tac_hybrid_flat_dim128(
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_values: PythonObject,
    py_centroid_values: PythonObject,
    py_centroid_doc_offsets: PythonObject,
    py_centroid_doc_indices: PythonObject,
) raises -> PythonObject:
    var index = decode_hybrid_flat_dim128_index(
        py_doc_ids, py_doc_offsets, py_token_values
    )
    return PythonObject(
        alloc=prepare_tachiom_tac_hybrid_flat_dim128_index(
            index^,
            decode_float_values(py_centroid_values),
            decode_int_list(py_centroid_doc_offsets),
            decode_int_list(py_centroid_doc_indices),
        )
    )


def prepare_tachiom_tac_i8_hybrid_flat_dim128(
    py_doc_ids: PythonObject,
    py_doc_offsets: PythonObject,
    py_token_values: PythonObject,
    py_centroid_values: PythonObject,
    py_centroid_doc_offsets: PythonObject,
    py_centroid_doc_indices: PythonObject,
) raises -> PythonObject:
    var index = decode_hybrid_flat_dim128_index(
        py_doc_ids, py_doc_offsets, py_token_values
    )
    return PythonObject(
        alloc=prepare_tachiom_tac_i8_hybrid_flat_dim128_index(
            index,
            decode_float_values(py_centroid_values),
            decode_int_list(py_centroid_doc_offsets),
            decode_int_list(py_centroid_doc_indices),
        )
    )


def prepare_tachiom_tac_hnsw_hybrid_flat_dim128(
    py_request: PythonObject,
) raises -> PythonObject:
    var index = decode_hybrid_flat_dim128_index(
        py_request[0], py_request[1], py_request[2]
    )
    return PythonObject(
        alloc=prepare_tachiom_tac_hnsw_hybrid_flat_dim128_index(
            index^,
            decode_float_values(py_request[3]),
            decode_int_list(py_request[4]),
            decode_int_list(py_request[5]),
            decode_int_list(py_request[6]),
            decode_int_list(py_request[7]),
            decode_int_list(py_request[8]),
            Int(py=py_request[9]),
        )
    )


def prepare_tachiom_tac_pq_dim128(
    py_request: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=prepare_tachiom_tac_pq_dim128_index(
            decode_string_list(py_request[0]),
            decode_int_list(py_request[1]),
            decode_float_values(py_request[2]),
            decode_int_list(py_request[3]),
            decode_int_list(py_request[4]),
            decode_int_list(py_request[5]),
            decode_float_values(py_request[6]),
            decode_int_list(py_request[7]),
            decode_float_values(py_request[8]),
            Int(py=py_request[9]),
            Int(py=py_request[10]),
        )
    )


def prepare_tachiom_tac_hnsw_pq_dim128(
    py_request: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=prepare_tachiom_tac_hnsw_pq_dim128_index(
            decode_string_list(py_request[0]),
            decode_int_list(py_request[1]),
            decode_float_values(py_request[2]),
            decode_int_list(py_request[3]),
            decode_int_list(py_request[4]),
            decode_int_list(py_request[5]),
            decode_float_values(py_request[6]),
            decode_int_list(py_request[7]),
            decode_float_values(py_request[8]),
            Int(py=py_request[9]),
            Int(py=py_request[10]),
            decode_int_list(py_request[11]),
            decode_int_list(py_request[12]),
            decode_int_list(py_request[13]),
            Int(py=py_request[14]),
        )
    )


def prepare_tachiom_tac_hnsw_pq_dim128_address(
    py_request: PythonObject,
) raises -> PythonObject:
    return PythonObject(
        alloc=prepare_tachiom_tac_hnsw_pq_dim128_address_index(
            Int(py=py_request[0]),
            Int(py=py_request[1]),
            Int(py=py_request[2]),
            Int(py=py_request[3]),
            Int(py=py_request[4]),
            Int(py=py_request[5]),
            Int(py=py_request[6]),
            Int(py=py_request[7]),
            Int(py=py_request[8]),
            Int(py=py_request[9]),
            Int(py=py_request[10]),
            Int(py=py_request[11]),
            Int(py=py_request[12]),
            Int(py=py_request[13]),
            Int(py=py_request[14]),
            Int(py=py_request[15]),
            Int(py=py_request[16]),
            Int(py=py_request[17]),
            Int(py=py_request[18]),
            Int(py=py_request[19]),
        )
    )


def plaid_approx_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxIndex
    ]()
    return Python.int(
        plaid_approx_prepared_posting_count_value(prepared_index[])
    )


def plaid_approx_i8_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return Python.int(
        plaid_approx_i8_prepared_posting_count_value(prepared_index[])
    )


def tachiom_tac_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacIndex
    ]()
    return Python.int(tachiom_tac_prepared_posting_count_value(prepared_index[]))


def tachiom_tac_i8_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacI8Index
    ]()
    return Python.int(tachiom_tac_i8_prepared_posting_count_value(prepared_index[]))


def tachiom_tac_hnsw_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacHnswIndex
    ]()
    return Python.int(tachiom_tac_hnsw_prepared_posting_count_value(prepared_index[]))


def tachiom_tac_hnsw_prepared_graph_edge_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacHnswIndex
    ]()
    return Python.int(tachiom_tac_hnsw_prepared_graph_edge_count_value(prepared_index[]))


def tachiom_tac_pq_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacPqIndex
    ]()
    return Python.int(tachiom_tac_pq_prepared_posting_count_value(prepared_index[]))


def tachiom_tac_hnsw_pq_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacHnswPqIndex
    ]()
    return Python.int(
        tachiom_tac_hnsw_pq_prepared_posting_count_value(prepared_index[])
    )


def tachiom_tac_hnsw_pq_prepared_graph_edge_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacHnswPqIndex
    ]()
    return Python.int(
        tachiom_tac_hnsw_pq_prepared_graph_edge_count_value(prepared_index[])
    )


def tachiom_tac_hnsw_pq_address_prepared_posting_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacHnswPqAddressIndex
    ]()
    return Python.int(
        tachiom_tac_hnsw_pq_address_prepared_posting_count_value(prepared_index[])
    )


def tachiom_tac_hnsw_pq_address_prepared_graph_edge_count(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacHnswPqAddressIndex
    ]()
    return Python.int(
        tachiom_tac_hnsw_pq_address_prepared_graph_edge_count_value(
            prepared_index[]
        )
    )


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


def plaid_i8_prepared_centroid_token_indices(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return int_values_to_python(prepared_index[].centroid_token_indices)


def plaid_i8_prepared_centroid_doc_offsets(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return int_values_to_python(prepared_index[].centroid_doc_offsets)


def plaid_i8_prepared_centroid_doc_indices(
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    return int_values_to_python(prepared_index[].centroid_doc_indices)


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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

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


def search_tachiom_tac_prepared_batch(
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
        PreparedTachiomTacIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedTachiomTacIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[6]))
    var prepared_index = py_request[7].downcast_value_ptr[
        PreparedTachiomTacIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_search_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_prepared_hits_batch(
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
        PreparedTachiomTacIndex
    ]()
    var hits_by_query = List[List[SearchHit]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        hits_by_query.append(
            tachiom_tac_search_hits_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return hits_batch_to_python(hits_by_query)


def tachiom_tac_candidate_positions_prepared_batch(
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
        PreparedTachiomTacIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_candidate_positions_prepared_batch_address(
    py_query_values_address: PythonObject,
    py_query_count: PythonObject,
    py_query_vector_count: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_query_values_address)
    var query_count = Int(py=py_query_count)
    var query_vector_count = Int(py=py_query_vector_count)
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_centroids_per_query_vector)
    var candidate_k = Int(py=py_candidate_k)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedTachiomTacIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_candidate_positions_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var final_k = Int(py=py_request[5])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[6]))
    var prepared_index = py_request[7].downcast_value_ptr[
        PreparedTachiomTacIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_candidate_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_i8_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedTachiomTacI8Index
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_i8_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_i8_candidate_positions_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var prepared_index = py_request[5].downcast_value_ptr[
        PreparedTachiomTacI8Index
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_i8_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_hnsw_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[7]))
    var prepared_index = py_request[8].downcast_value_ptr[
        PreparedTachiomTacHnswIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_hnsw_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_hnsw_candidate_positions_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var ef_search = Int(py=py_request[5])
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedTachiomTacHnswIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_hnsw_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                ef_search,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_hnsw_candidate_positions_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var final_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[7]))
    var prepared_index = py_request[8].downcast_value_ptr[
        PreparedTachiomTacHnswIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_hnsw_candidate_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_hnsw_pq_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var prepared_index = py_request[7].downcast_value_ptr[
        PreparedTachiomTacHnswPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].pq.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_hnsw_pq_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_hnsw_pq_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[7]))
    var prepared_index = py_request[8].downcast_value_ptr[
        PreparedTachiomTacHnswPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].pq.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_hnsw_pq_search_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var ef_search = Int(py=py_request[5])
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedTachiomTacHnswPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].pq.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_hnsw_pq_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                ef_search,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var final_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[7]))
    var prepared_index = py_request[8].downcast_value_ptr[
        PreparedTachiomTacHnswPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].pq.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_hnsw_pq_candidate_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_hnsw_pq_address_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var prepared_index = py_request[7].downcast_value_ptr[
        PreparedTachiomTacHnswPqAddressIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != 128:
            raise Error("all queries must be dim128")

        positions_by_query.append(
            tachiom_tac_hnsw_pq_address_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_hnsw_pq_address_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[7]))
    var prepared_index = py_request[8].downcast_value_ptr[
        PreparedTachiomTacHnswPqAddressIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != 128:
            raise Error("all queries must be dim128")

        positions_by_query.append(
            tachiom_tac_hnsw_pq_address_search_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var ef_search = Int(py=py_request[5])
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedTachiomTacHnswPqAddressIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != 128:
            raise Error("all queries must be dim128")

        positions_by_query.append(
            tachiom_tac_hnsw_pq_address_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                ef_search,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var final_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[7]))
    var prepared_index = py_request[8].downcast_value_ptr[
        PreparedTachiomTacHnswPqAddressIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != 128:
            raise Error("all queries must be dim128")

        positions_by_query.append(
            tachiom_tac_hnsw_pq_address_candidate_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                ef_search,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_pq_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedTachiomTacPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_pq_search_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def search_tachiom_tac_pq_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var final_k = Int(py=py_request[3])
    var centroids_per_query_vector = Int(py=py_request[4])
    var candidate_k = Int(py=py_request[5])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[6]))
    var prepared_index = py_request[7].downcast_value_ptr[
        PreparedTachiomTacPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_pq_search_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_pq_candidate_positions_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var prepared_index = py_request[5].downcast_value_ptr[
        PreparedTachiomTacPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_pq_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def tachiom_tac_pq_candidate_positions_prepared_batch_address_with_pruning(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var final_k = Int(py=py_request[5])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[6]))
    var prepared_index = py_request[7].downcast_value_ptr[
        PreparedTachiomTacPqIndex
    ]()
    var positions_by_query = List[List[Int]]()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            tachiom_tac_pq_candidate_positions_for_query_with_pruning(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
                final_k,
                candidate_pruning_alpha,
            )
        )

    return positions_batch_to_python(positions_by_query)


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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            plaid_i8_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def plaid_i8_candidate_positions_prepared_batch_address(
    py_query_values_address: PythonObject,
    py_query_count: PythonObject,
    py_query_vector_count: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_query_values_address)
    var query_count = Int(py=py_query_count)
    var query_vector_count = Int(py=py_query_vector_count)
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            plaid_i8_candidate_positions_for_query(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def plaid_i8_candidate_positions_prepared_batch_address_unordered(
    py_query_values_address: PythonObject,
    py_query_count: PythonObject,
    py_query_vector_count: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_query_values_address)
    var query_count = Int(py=py_query_count)
    var query_vector_count = Int(py=py_query_vector_count)
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            plaid_i8_candidate_positions_for_query_unordered(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def plaid_i8_positive_centroid_candidate_positions_prepared_batch_address_unordered(
    py_query_values_address: PythonObject,
    py_query_count: PythonObject,
    py_query_vector_count: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_candidate_k: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_query_values_address)
    var query_count = Int(py=py_query_count)
    var query_vector_count = Int(py=py_query_vector_count)
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        positions_by_query.append(
            plaid_i8_positive_centroid_candidate_positions_for_query_unordered(
                query,
                prepared_index[],
                centroids_per_query_vector,
                candidate_k,
            )
        )

    return positions_batch_to_python(positions_by_query)


def plaid_i8_selected_centroids_prepared_batch_address(
    py_query_values_address: PythonObject,
    py_query_count: PythonObject,
    py_query_vector_count: PythonObject,
    py_centroids_per_query_vector: PythonObject,
    py_prepared_index: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_query_values_address)
    var query_count = Int(py=py_query_count)
    var query_vector_count = Int(py=py_query_vector_count)
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        var empty = Python.list()
        empty.append(Python.list())
        empty.append(Python.list())
        return empty

    var centroids_per_query_vector = Int(py=py_centroids_per_query_vector)
    var prepared_index = py_prepared_index.downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    var positions_by_query = Python.list()
    var scores_by_query = Python.list()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        var centroid_scores_by_query_vector = (
            build_i8_centroid_scores_by_query_vector(query, prepared_index[])
        )
        var centroid_positions_by_query_vector = (
            build_i8_centroid_positions_by_query_vector(
                centroid_scores_by_query_vector,
                centroids_per_query_vector,
            )
        )
        var query_positions = Python.list()
        var query_scores = Python.list()

        for query_vector_index in range(
            len(centroid_positions_by_query_vector)
        ):
            for selected_offset in range(
                len(centroid_positions_by_query_vector[query_vector_index])
            ):
                var centroid_position = centroid_positions_by_query_vector[
                    query_vector_index
                ][selected_offset]
                query_positions.append(Python.int(centroid_position))
                query_scores.append(
                    Python.float(
                        centroid_scores_by_query_vector[query_vector_index][
                            centroid_position
                        ]
                    )
                )

        positions_by_query.append(query_positions)
        scores_by_query.append(query_scores)

    var result = Python.list()
    result.append(positions_by_query)
    result.append(scores_by_query)
    return result


def plaid_i8_selected_centroids_prepared_batch_address_into(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var centroids_per_query_vector = Int(py=py_request[3])
    var positions_address = Int(py=py_request[4])
    var scores_address = Int(py=py_request[5])
    if positions_address == 0:
        raise Error("selected centroid positions address must be non-zero")
    if scores_address == 0:
        raise Error("selected centroid scores address must be non-zero")
    if centroids_per_query_vector <= 0:
        raise Error("centroids_per_query_vector must be positive")

    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    var selected_per_query = query_vector_count * centroids_per_query_vector
    var positions_out = UnsafePointer[Int64, MutAnyOrigin](
        unsafe_from_address=positions_address
    )
    var scores_out = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=scores_address
    )

    for query_index in range(len(queries)):
        var query = queries[query_index].copy()
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

        var centroid_scores_by_query_vector = (
            build_i8_centroid_scores_by_query_vector(query, prepared_index[])
        )
        var centroid_positions_by_query_vector = (
            build_i8_centroid_positions_by_query_vector(
                centroid_scores_by_query_vector,
                centroids_per_query_vector,
            )
        )
        var query_base = query_index * selected_per_query
        for query_vector_index in range(
            len(centroid_positions_by_query_vector)
        ):
            var vector_base = (
                query_base + query_vector_index * centroids_per_query_vector
            )
            for selected_offset in range(
                len(centroid_positions_by_query_vector[query_vector_index])
            ):
                var centroid_position = centroid_positions_by_query_vector[
                    query_vector_index
                ][selected_offset]
                var out_index = vector_base + selected_offset
                positions_out[out_index] = Int64(centroid_position)
                scores_out[out_index] = Float32(
                    centroid_scores_by_query_vector[query_vector_index][
                        centroid_position
                    ]
                )

    return Python.int(query_count * selected_per_query)


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
            raise Error(
                "all queries must share the prepared index vector dimension"
            )

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


def plaid_i8_candidate_generation_profile_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var measurement_iterations = Int(py=py_request[5])
    var prepared_index = py_request[6].downcast_value_ptr[
        PreparedPlaidApproxI8Index
    ]()
    var py_profiles = Python.list()

    for query in queries:
        if query.vector_dim != prepared_index[].vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )
        var profile = profile_plaid_i8_candidate_generation_for_query(
            query,
            prepared_index[],
            centroids_per_query_vector,
            candidate_k,
            measurement_iterations,
        )
        py_profiles.append(i8_candidate_generation_profile_to_python(profile))

    return py_profiles


def tachiom_tac_candidate_generation_profile_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var final_k = Int(py=py_request[5])
    var measurement_iterations = Int(py=py_request[6])
    var prepared_index = py_request[7].downcast_value_ptr[
        PreparedTachiomTacIndex
    ]()
    var py_profiles = Python.list()

    for query in queries:
        if query.vector_dim != prepared_index[].index.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )
        var profile = profile_tachiom_tac_candidate_generation_for_query(
            query,
            prepared_index[],
            centroids_per_query_vector,
            candidate_k,
            final_k,
            measurement_iterations,
        )
        py_profiles.append(tachiom_tac_candidate_generation_profile_to_python(profile))

    return py_profiles


def tachiom_tac_hnsw_pq_query_profile_prepared_batch_address(
    py_request: PythonObject,
) raises -> PythonObject:
    var query_values_address = Int(py=py_request[0])
    var query_count = Int(py=py_request[1])
    var query_vector_count = Int(py=py_request[2])
    var queries = decode_flat_queries_from_float32_address(
        query_values_address,
        query_count,
        query_vector_count,
    )
    if len(queries) == 0:
        return Python.list()

    var centroids_per_query_vector = Int(py=py_request[3])
    var candidate_k = Int(py=py_request[4])
    var final_k = Int(py=py_request[5])
    var ef_search = Int(py=py_request[6])
    var candidate_pruning_alpha = ScoreScalar(Float64(py=py_request[7]))
    var measurement_iterations = Int(py=py_request[8])
    var prepared_index = py_request[9].downcast_value_ptr[
        PreparedTachiomTacHnswPqIndex
    ]()
    var py_profiles = Python.list()

    for query in queries:
        if query.vector_dim != prepared_index[].pq.vector_dim:
            raise Error(
                "all queries must share the prepared index vector dimension"
            )
        var profile = profile_tachiom_tac_hnsw_pq_for_query(
            query,
            prepared_index[],
            centroids_per_query_vector,
            candidate_k,
            final_k,
            ef_search,
            candidate_pruning_alpha,
            measurement_iterations,
        )
        py_profiles.append(tachiom_tac_hnsw_pq_query_profile_to_python(profile))

    return py_profiles


@export
def PyInit__mojo_exact_cpu_bindings() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojo_exact_cpu_bindings")
        _ = module.add_type[PreparedPackedIndex]("PreparedPackedIndex")
        _ = module.add_type[PreparedPlaidApproxIndex](
            "PreparedPlaidApproxIndex"
        )
        _ = module.add_type[PreparedPlaidApproxI8Index](
            "PreparedPlaidApproxI8Index"
        )
        _ = module.add_type[PreparedTachiomTacI8Index](
            "PreparedTachiomTacI8Index"
        )
        _ = module.add_type[PreparedTachiomTacHnswIndex](
            "PreparedTachiomTacHnswIndex"
        )
        _ = module.add_type[PreparedTachiomTacHnswPqIndex](
            "PreparedTachiomTacHnswPqIndex"
        )
        _ = module.add_type[PreparedTachiomTacHnswPqAddressIndex](
            "PreparedTachiomTacHnswPqAddressIndex"
        )
        _ = module.add_type[PreparedTachiomTacIndex](
            "PreparedTachiomTacIndex"
        )
        _ = module.add_type[PreparedTachiomTacPqIndex](
            "PreparedTachiomTacPqIndex"
        )
        module.def_function[exact_scores_packed](
            "exact_scores_packed",
            docstring=(
                "Score a packed late-interaction index with the exact CPU"
                " backend."
            ),
        )
        module.def_function[prepare_packed_index](
            "prepare_packed_index",
            docstring=(
                "Decode and retain a packed index inside a Mojo-backed Python"
                " object."
            ),
        )
        module.def_function[prepare_packed_index_from_storage](
            "prepare_packed_index_from_storage",
            docstring="Load and retain a packed index from Kayak storage.",
        )
        module.def_function[prepare_plaid_approx_hybrid_flat_dim128](
            "prepare_plaid_approx_hybrid_flat_dim128",
            docstring=(
                "Prepare a sampled-centroid PLAID-style approximation index in"
                " Mojo."
            ),
        )
        module.def_function[prepare_plaid_approx_i8_hybrid_flat_dim128](
            "prepare_plaid_approx_i8_hybrid_flat_dim128",
            docstring=(
                "Prepare an int8 sampled-centroid PLAID-style approximation"
                " index in Mojo."
            ),
        )
        module.def_function[prepare_tachiom_tac_hybrid_flat_dim128](
            "prepare_tachiom_tac_hybrid_flat_dim128",
            docstring=(
                "Prepare a Tachiom TAC index from token-aware centroids in Mojo."
            ),
        )
        module.def_function[prepare_tachiom_tac_i8_hybrid_flat_dim128](
            "prepare_tachiom_tac_i8_hybrid_flat_dim128",
            docstring=(
                "Prepare a Tachiom TAC index with an int8 rerank payload in Mojo."
            ),
        )
        module.def_function[prepare_tachiom_tac_hnsw_hybrid_flat_dim128](
            "prepare_tachiom_tac_hnsw_hybrid_flat_dim128",
            docstring=(
                "Prepare a Tachiom TAC index with an HNSW centroid graph in Mojo."
            ),
        )
        module.def_function[prepare_tachiom_tac_pq_dim128](
            "prepare_tachiom_tac_pq_dim128",
            docstring=(
                "Prepare a Tachiom TAC index with a residual-PQ rerank payload"
                " in Mojo."
            ),
        )
        module.def_function[prepare_tachiom_tac_hnsw_pq_dim128](
            "prepare_tachiom_tac_hnsw_pq_dim128",
            docstring=(
                "Prepare a Tachiom TAC HNSW index with a residual-PQ rerank"
                " payload in Mojo."
            ),
        )
        module.def_function[prepare_tachiom_tac_hnsw_pq_dim128_address](
            "prepare_tachiom_tac_hnsw_pq_dim128_address",
            docstring=(
                "Prepare an address-backed streaming Tachiom TAC HNSW"
                " residual-PQ index in Mojo."
            ),
        )
        module.def_function[plaid_approx_prepared_posting_count](
            "plaid_approx_prepared_posting_count",
            docstring="Return the prepared PLAID-style index posting count.",
        )
        module.def_function[plaid_approx_i8_prepared_posting_count](
            "plaid_approx_i8_prepared_posting_count",
            docstring=(
                "Return the prepared int8 PLAID-style index posting count."
            ),
        )
        module.def_function[tachiom_tac_prepared_posting_count](
            "tachiom_tac_prepared_posting_count",
            docstring="Return the prepared Tachiom TAC posting count.",
        )
        module.def_function[tachiom_tac_i8_prepared_posting_count](
            "tachiom_tac_i8_prepared_posting_count",
            docstring="Return the prepared Tachiom TAC i8 posting count.",
        )
        module.def_function[tachiom_tac_hnsw_prepared_posting_count](
            "tachiom_tac_hnsw_prepared_posting_count",
            docstring="Return the prepared Tachiom TAC HNSW posting count.",
        )
        module.def_function[tachiom_tac_hnsw_prepared_graph_edge_count](
            "tachiom_tac_hnsw_prepared_graph_edge_count",
            docstring="Return the prepared Tachiom TAC HNSW graph edge count.",
        )
        module.def_function[tachiom_tac_pq_prepared_posting_count](
            "tachiom_tac_pq_prepared_posting_count",
            docstring="Return the prepared Tachiom TAC residual-PQ posting count.",
        )
        module.def_function[tachiom_tac_hnsw_pq_prepared_posting_count](
            "tachiom_tac_hnsw_pq_prepared_posting_count",
            docstring=(
                "Return the prepared Tachiom TAC HNSW residual-PQ posting count."
            ),
        )
        module.def_function[tachiom_tac_hnsw_pq_prepared_graph_edge_count](
            "tachiom_tac_hnsw_pq_prepared_graph_edge_count",
            docstring=(
                "Return the prepared Tachiom TAC HNSW residual-PQ graph edge count."
            ),
        )
        module.def_function[tachiom_tac_hnsw_pq_address_prepared_posting_count](
            "tachiom_tac_hnsw_pq_address_prepared_posting_count",
            docstring=(
                "Return the address-backed Tachiom TAC HNSW residual-PQ posting count."
            ),
        )
        module.def_function[tachiom_tac_hnsw_pq_address_prepared_graph_edge_count](
            "tachiom_tac_hnsw_pq_address_prepared_graph_edge_count",
            docstring=(
                "Return the address-backed Tachiom TAC HNSW residual-PQ graph edge count."
            ),
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
        module.def_function[plaid_i8_prepared_centroid_token_indices](
            "plaid_i8_prepared_centroid_token_indices",
            docstring="Return prepared int8 PLAID centroid token indices.",
        )
        module.def_function[plaid_i8_prepared_centroid_doc_offsets](
            "plaid_i8_prepared_centroid_doc_offsets",
            docstring="Return prepared int8 PLAID centroid posting offsets.",
        )
        module.def_function[plaid_i8_prepared_centroid_doc_indices](
            "plaid_i8_prepared_centroid_doc_indices",
            docstring="Return prepared int8 PLAID centroid posting indices.",
        )
        module.def_function[exact_scores_packed_batch](
            "exact_scores_packed_batch",
            docstring=(
                "Score a packed late-interaction index for a whole batch of"
                " queries."
            ),
        )
        module.def_function[exact_scores_prepared_packed](
            "exact_scores_prepared_packed",
            docstring="Score a prepared packed late-interaction index.",
        )
        module.def_function[exact_scores_prepared_packed_batch](
            "exact_scores_prepared_packed_batch",
            docstring=(
                "Score a prepared packed late-interaction index for a whole"
                " batch of queries."
            ),
        )
        module.def_function[search_prepared_packed](
            "search_prepared_packed",
            docstring="Search a prepared packed late-interaction index.",
        )
        module.def_function[search_prepared_packed_batch](
            "search_prepared_packed_batch",
            docstring=(
                "Search a prepared packed late-interaction index for a whole"
                " batch of queries."
            ),
        )
        module.def_function[exact_scores_hybrid_flat_dim128](
            "exact_scores_hybrid_flat_dim128",
            docstring="Score a hybrid flat dim128 index with a nested query.",
        )
        module.def_function[exact_scores_hybrid_flat_dim128_with_flat_query](
            "exact_scores_hybrid_flat_dim128_with_flat_query",
            docstring=(
                "Score a hybrid flat dim128 index with a flat dim128 query."
            ),
        )
        module.def_function[search_plaid_approx_prepared_batch](
            "search_plaid_approx_prepared_batch",
            docstring=(
                "Search a prepared sampled-centroid PLAID-style approximation"
                " index."
            ),
        )
        module.def_function[search_plaid_approx_prepared_hits_batch](
            "search_plaid_approx_prepared_hits_batch",
            docstring=(
                "Search a prepared sampled-centroid PLAID-style approximation"
                " index and return hits."
            ),
        )
        module.def_function[search_plaid_approx_i8_prepared_batch](
            "search_plaid_approx_i8_prepared_batch",
            docstring=(
                "Search a prepared int8 sampled-centroid PLAID-style"
                " approximation index."
            ),
        )
        module.def_function[search_plaid_approx_i8_prepared_hits_batch](
            "search_plaid_approx_i8_prepared_hits_batch",
            docstring=(
                "Search a prepared int8 sampled-centroid PLAID-style"
                " approximation index and return hits."
            ),
        )
        module.def_function[search_tachiom_tac_prepared_batch](
            "search_tachiom_tac_prepared_batch",
            docstring="Search a prepared Tachiom TAC index.",
        )
        module.def_function[search_tachiom_tac_prepared_batch_address](
            "search_tachiom_tac_prepared_batch_address",
            docstring=(
                "Search a prepared Tachiom TAC index from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[
            search_tachiom_tac_prepared_batch_address_with_pruning
        ](
            "search_tachiom_tac_prepared_batch_address_with_pruning",
            docstring=(
                "Search a prepared Tachiom TAC index with candidate pruning"
                " from a contiguous float32 query tensor address."
            ),
        )
        module.def_function[search_tachiom_tac_prepared_hits_batch](
            "search_tachiom_tac_prepared_hits_batch",
            docstring="Search a prepared Tachiom TAC index and return hits.",
        )
        module.def_function[tachiom_tac_candidate_positions_prepared_batch](
            "tachiom_tac_candidate_positions_prepared_batch",
            docstring="Generate Tachiom TAC candidate document positions.",
        )
        module.def_function[
            tachiom_tac_candidate_positions_prepared_batch_address
        ](
            "tachiom_tac_candidate_positions_prepared_batch_address",
            docstring=(
                "Generate Tachiom TAC candidates from a contiguous float32"
                " query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_candidate_positions_prepared_batch_address_with_pruning
        ](
            "tachiom_tac_candidate_positions_prepared_batch_address_with_pruning",
            docstring=(
                "Generate pruned Tachiom TAC candidates from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[search_tachiom_tac_i8_prepared_batch_address](
            "search_tachiom_tac_i8_prepared_batch_address",
            docstring=(
                "Search a prepared Tachiom TAC i8 index from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_i8_candidate_positions_prepared_batch_address
        ](
            "tachiom_tac_i8_candidate_positions_prepared_batch_address",
            docstring=(
                "Generate Tachiom TAC i8 candidates from a contiguous float32"
                " query tensor address."
            ),
        )
        module.def_function[search_tachiom_tac_hnsw_prepared_batch_address](
            "search_tachiom_tac_hnsw_prepared_batch_address",
            docstring=(
                "Search a prepared Tachiom TAC HNSW index from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_hnsw_candidate_positions_prepared_batch_address
        ](
            "tachiom_tac_hnsw_candidate_positions_prepared_batch_address",
            docstring=(
                "Generate Tachiom TAC HNSW candidates from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_hnsw_candidate_positions_prepared_batch_address_with_pruning
        ](
            "tachiom_tac_hnsw_candidate_positions_prepared_batch_address_with_pruning",
            docstring=(
                "Generate pruned Tachiom TAC HNSW candidates from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[
            search_tachiom_tac_hnsw_pq_prepared_batch_address
        ](
            "search_tachiom_tac_hnsw_pq_prepared_batch_address",
            docstring=(
                "Search a prepared Tachiom TAC HNSW residual-PQ index from a"
                " contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            search_tachiom_tac_hnsw_pq_prepared_batch_address_with_pruning
        ](
            "search_tachiom_tac_hnsw_pq_prepared_batch_address_with_pruning",
            docstring=(
                "Search a prepared Tachiom TAC HNSW residual-PQ index with"
                " candidate pruning from a contiguous float32 query tensor"
                " address."
            ),
        )
        module.def_function[
            tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address
        ](
            "tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address",
            docstring=(
                "Generate Tachiom TAC HNSW residual-PQ candidates from a"
                " contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address_with_pruning
        ](
            "tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address_with_pruning",
            docstring=(
                "Generate pruned Tachiom TAC HNSW residual-PQ candidates from"
                " a contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            search_tachiom_tac_hnsw_pq_address_prepared_batch_address
        ](
            "search_tachiom_tac_hnsw_pq_address_prepared_batch_address",
            docstring=(
                "Search an address-backed Tachiom TAC HNSW residual-PQ index"
                " from a contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            search_tachiom_tac_hnsw_pq_address_prepared_batch_address_with_pruning
        ](
            "search_tachiom_tac_hnsw_pq_address_prepared_batch_address_with_pruning",
            docstring=(
                "Search an address-backed Tachiom TAC HNSW residual-PQ index"
                " with pruning from a contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address
        ](
            "tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address",
            docstring=(
                "Generate address-backed Tachiom TAC HNSW residual-PQ"
                " candidates from a contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address_with_pruning
        ](
            "tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address_with_pruning",
            docstring=(
                "Generate pruned address-backed Tachiom TAC HNSW residual-PQ"
                " candidates from a contiguous float32 query tensor address."
            ),
        )
        module.def_function[search_tachiom_tac_pq_prepared_batch_address](
            "search_tachiom_tac_pq_prepared_batch_address",
            docstring=(
                "Search a prepared Tachiom TAC residual-PQ index from a"
                " contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            search_tachiom_tac_pq_prepared_batch_address_with_pruning
        ](
            "search_tachiom_tac_pq_prepared_batch_address_with_pruning",
            docstring=(
                "Search a prepared Tachiom TAC residual-PQ index with"
                " candidate pruning from a contiguous float32 query tensor"
                " address."
            ),
        )
        module.def_function[
            tachiom_tac_pq_candidate_positions_prepared_batch_address
        ](
            "tachiom_tac_pq_candidate_positions_prepared_batch_address",
            docstring=(
                "Generate Tachiom TAC residual-PQ candidates from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_pq_candidate_positions_prepared_batch_address_with_pruning
        ](
            "tachiom_tac_pq_candidate_positions_prepared_batch_address_with_pruning",
            docstring=(
                "Generate pruned Tachiom TAC residual-PQ candidates from a"
                " contiguous float32 query tensor address."
            ),
        )
        module.def_function[plaid_i8_candidate_positions_prepared_batch](
            "plaid_i8_candidate_positions_prepared_batch",
            docstring=(
                "Generate int8 PLAID candidate document positions without"
                " reranking."
            ),
        )
        module.def_function[
            plaid_i8_candidate_positions_prepared_batch_address
        ](
            "plaid_i8_candidate_positions_prepared_batch_address",
            docstring=(
                "Generate int8 PLAID candidate positions from a contiguous"
                " float32 query tensor address."
            ),
        )
        module.def_function[
            plaid_i8_candidate_positions_prepared_batch_address_unordered
        ](
            "plaid_i8_candidate_positions_prepared_batch_address_unordered",
            docstring=(
                "Generate unordered int8 PLAID candidate positions from a"
                " contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            plaid_i8_positive_centroid_candidate_positions_prepared_batch_address_unordered
        ](
            "plaid_i8_positive_centroid_candidate_positions_prepared_batch_address_unordered",
            docstring=(
                "Generate unordered int8 PLAID candidate positions from"
                " positive selected centroid postings only."
            ),
        )
        module.def_function[plaid_i8_candidate_scores_prepared_batch](
            "plaid_i8_candidate_scores_prepared_batch",
            docstring=(
                "Score provided candidate positions with the prepared int8"
                " reranker."
            ),
        )
        module.def_function[plaid_i8_selected_centroids_prepared_batch_address](
            "plaid_i8_selected_centroids_prepared_batch_address",
            docstring=(
                "Export selected int8 PLAID centroid positions and proxy"
                " scores from a contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            plaid_i8_selected_centroids_prepared_batch_address_into
        ](
            "plaid_i8_selected_centroids_prepared_batch_address_into",
            docstring=(
                "Fill selected int8 PLAID centroid position and score arrays"
                " from a contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            plaid_i8_candidate_generation_profile_prepared_batch_address
        ](
            "plaid_i8_candidate_generation_profile_prepared_batch_address",
            docstring=(
                "Profile int8 PLAID candidate-generation substeps from a"
                " contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_candidate_generation_profile_prepared_batch_address
        ](
            "tachiom_tac_candidate_generation_profile_prepared_batch_address",
            docstring=(
                "Profile Tachiom TAC candidate-generation substeps from a"
                " contiguous float32 query tensor address."
            ),
        )
        module.def_function[
            tachiom_tac_hnsw_pq_query_profile_prepared_batch_address
        ](
            "tachiom_tac_hnsw_pq_query_profile_prepared_batch_address",
            docstring=(
                "Profile native Tachiom TAC HNSW residual-PQ search substeps"
                " from a contiguous float32 query tensor address."
            ),
        )
        return module.finalize()
    except e:
        abort(String("error creating kayak mojo exact cpu bindings:", e))
