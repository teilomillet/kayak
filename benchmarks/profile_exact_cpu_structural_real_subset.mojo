import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List

from kayak.contracts import EncodedQuery
from kayak import JudgedTask, PackedIndex, ScoreScalar
from kayak.numeric import VectorScalar, min_score_scalar
from kayak.scoring.maxsim import exact_score_for_document_dim128
from kayak.storage import (
    ensure_fiqa_real_subset_cache,
    ensure_scifact_real_subset_cache,
)
from benchmarks.structural.flat_dim128 import dot_product_dim128_flat_at


struct FlatPackedIndex(Copyable):
    var doc_ids: List[String]
    var doc_offsets: List[Int]
    var token_values: List[VectorScalar]
    var document_count: Int

    def __init__(
        out self,
        var doc_ids: List[String],
        var doc_offsets: List[Int],
        var token_values: List[VectorScalar],
        document_count: Int,
    ):
        self.doc_ids = doc_ids^
        self.doc_offsets = doc_offsets^
        self.token_values = token_values^
        self.document_count = document_count


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def flatten_index(read index: PackedIndex) -> FlatPackedIndex:
    var token_values = List[VectorScalar]()

    for token in index.token_vectors:
        for value in token:
            token_values.append(value)

    return FlatPackedIndex(
        index.doc_ids.copy(),
        index.doc_offsets.copy(),
        token_values^,
        index.document_count,
    )


def exact_scores_for_flat_index_dim128(
    read query: EncodedQuery, read index: FlatPackedIndex
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()

    for document_index in range(index.document_count):
        var start_vector = index.doc_offsets[document_index]
        var stop_vector = index.doc_offsets[document_index + 1]
        var start_offset = start_vector * 128
        var document_vector_count = stop_vector - start_vector
        var total = ScoreScalar(0.0)

        for query_token in query.token_vectors:
            var best_similarity = min_score_scalar()
            for token_index in range(document_vector_count):
                var similarity = dot_product_dim128_flat_at(
                    query_token,
                    index.token_values,
                    start_offset + (token_index * 128),
                )
                if similarity > best_similarity:
                    best_similarity = similarity
            total += best_similarity

        scores.append(total)
    return scores^


def exact_scores_for_nested_index_dim128(
    read query: EncodedQuery, read index: PackedIndex
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for document_index in range(index.document_count):
        scores.append(exact_score_for_document_dim128(query, index, document_index))
    return scores^


def benchmark_task(
    dataset_name: String,
    slice_name: String,
    task_source: String,
    index_source: String,
    read task: JudgedTask,
    read nested_index: PackedIndex,
    read flat_index: FlatPackedIndex,
) raises:
    print("dataset: ", dataset_name)
    print("slice: ", slice_name)
    print("task_source: ", task_source)
    print("index_source: ", index_source)
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)

    var query_index = 0
    print("layout: nested_serial_dim128")

    def nested_once() capturing:
        bench_compiler.keep(
            exact_scores_for_nested_index_dim128(
                task.queries[query_index].query, nested_index
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[nested_once]().print()
    print("")

    query_index = 0
    print("layout: flat_serial_dim128")

    def flat_once() capturing:
        bench_compiler.keep(
            exact_scores_for_flat_index_dim128(
                task.queries[query_index].query, flat_index
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[flat_once]().print()
    print("")


def benchmark_scifact_real_subset() raises:
    print("loading real BEIR/SciFact subset with storage...")
    var cache = ensure_scifact_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var nested_index = cache.stored_index.index.copy()
    var flat_index = flatten_index(nested_index)
    benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        source_label(cache.loaded_task_from_storage, "colbert_cpu_encode"),
        source_label(cache.loaded_index_from_storage, "pack_documents"),
        task,
        nested_index,
        flat_index,
    )


def benchmark_fiqa_real_subset() raises:
    print("loading real BEIR/FIQA subset with storage...")
    var cache = ensure_fiqa_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var nested_index = cache.stored_index.index.copy()
    var flat_index = flatten_index(nested_index)
    benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        source_label(cache.loaded_task_from_storage, "colbert_cpu_encode"),
        source_label(cache.loaded_index_from_storage, "pack_documents"),
        task,
        nested_index,
        flat_index,
    )


def main() raises:
    benchmark_scifact_real_subset()
    benchmark_fiqa_real_subset()
