import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    JudgedTask,
    NamespaceId,
    ScoredCentroidSelection,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_one_segment_collection_mirror,
    ensure_scifact_real_subset_cache,
    load_resolved_collection_snapshot,
    loaded_segment_stored_centroid_postings_index,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.index import CentroidPostingIndex
from kayak.numeric import ScoreScalar, VectorScalar
from kayak.planning.centroid_postings_imputed_stage import (
    centroid_selection_for_query_token,
    effective_imputed_centroid_bound,
    finalize_imputed_centroid_selection,
)
from kayak.planning.centroid_postings_stage import insert_descending_centroid_match
from kayak.scoring.dot import dot_product
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM


comptime CENTROID_HEAD_POSTING_CAP = 16


struct KernelMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var primitive_kind: String
    var vector_dim: Int
    var centroid_count: Int
    var query_count: Int
    var total_query_vectors: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var primitive_kind: String,
        vector_dim: Int,
        centroid_count: Int,
        query_count: Int,
        total_query_vectors: Int,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.primitive_kind = primitive_kind^
        self.vector_dim = vector_dim
        self.centroid_count = centroid_count
        self.query_count = query_count
        self.total_query_vectors = total_query_vectors
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(mut out: String, read measurement: KernelMeasurement):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.primitive_kind
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.centroid_count)
    out += "\t"
    out += String(measurement.query_count)
    out += "\t"
    out += String(measurement.total_query_vectors)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(path: Path, measurements: List[KernelMeasurement]) raises:
    var lines = String()
    lines += (
        "dataset_name\tslice_name\tprimitive_kind\tvector_dim\tcentroid_count"
        "\tquery_count\ttotal_query_vectors\tmean_seconds\n"
    )

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def load_profile_snapshot(
    collection_name: String,
    collection_root: Path,
    read stored_index: StoredPackedIndex,
) raises -> ResolvedCollectionSnapshot:
    var root = ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId(collection_name),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        stored_index,
        0,
        0,
        CENTROID_HEAD_POSTING_CAP,
    )
    return load_resolved_collection_snapshot(root, SnapshotId("snapshot-0001"))


def total_query_vectors(read task: JudgedTask) -> Int:
    var total = 0
    for judged_query in task.queries:
        total += judged_query.query.vector_count
    return total


def reference_centroid_selection_for_query_token_dim128(
    read query_token: List[VectorScalar],
    read index: CentroidPostingIndex,
    final_k: Int,
) -> ScoredCentroidSelection:
    var bound = effective_imputed_centroid_bound(index.centroid_count)
    var sorted_centroid_indices = List[Int]()
    var sorted_centroid_scores = List[ScoreScalar]()

    for centroid_index in range(index.centroid_count):
        insert_descending_centroid_match(
            sorted_centroid_indices,
            sorted_centroid_scores,
            centroid_index,
            dot_product(query_token, index.centroid_vectors[centroid_index]),
            bound,
        )

    return finalize_imputed_centroid_selection(
        sorted_centroid_indices^,
        sorted_centroid_scores^,
        index,
        final_k,
    )


def reference_nested_selections_for_query(
    read snapshot: ResolvedCollectionSnapshot,
    read task: JudgedTask,
    query_index: Int,
) raises -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    for token in task.queries[query_index].query.token_vectors:
        selections.append(
            reference_centroid_selection_for_query_token_dim128(
                token,
                index,
                task.k,
            )
        )

    return selections^


def optimized_nested_selections_for_query(
    read snapshot: ResolvedCollectionSnapshot,
    read task: JudgedTask,
    query_index: Int,
) raises -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    for token in task.queries[query_index].query.token_vectors:
        selections.append(centroid_selection_for_query_token(token, index, task.k))

    return selections^


def append_measurements_for_dataset(
    mut measurements: List[KernelMeasurement],
    dataset_name: String,
    read snapshot: ResolvedCollectionSnapshot,
    read task: JudgedTask,
) raises:
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()
    if index.vector_dim != COLBERT_VECTOR_DIM:
        raise Error("dim128 nested kernel benchmark requires vector_dim == 128")

    var total_vectors = total_query_vectors(task)
    print(
        "dataset= ",
        dataset_name,
        " slice= ",
        task.slice_name,
        " vector_dim= ",
        index.vector_dim,
        " centroid_count= ",
        index.centroid_count,
        " query_count= ",
        len(task.queries),
        " total_query_vectors= ",
        total_vectors,
    )
    print("")

    # Warm both paths before timing so the comparison reflects the selector
    # kernels rather than first-use compilation or cache effects.
    _ = reference_nested_selections_for_query(snapshot, task, 0)
    _ = optimized_nested_selections_for_query(snapshot, task, 0)

    var query_index = 0
    print("== imputed_selection_nested_reference ==")

    def reference_once() capturing raises:
        bench_compiler.keep(
            reference_nested_selections_for_query(snapshot, task, query_index)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[reference_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    report.print()
    print("")
    measurements.append(
        KernelMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "imputed_selection_nested_reference",
            index.vector_dim,
            index.centroid_count,
            len(task.queries),
            total_vectors,
            report.mean(),
        )
    )

    query_index = 0
    print("== imputed_selection_nested_optimized ==")

    def optimized_once() capturing raises:
        bench_compiler.keep(
            optimized_nested_selections_for_query(snapshot, task, query_index)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    report = benchmark.run[optimized_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    report.print()
    print("")
    measurements.append(
        KernelMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "imputed_selection_nested_optimized",
            index.vector_dim,
            index.centroid_count,
            len(task.queries),
            total_vectors,
            report.mean(),
        )
    )


def main() raises:
    var measurements = List[KernelMeasurement]()

    var scifact_cache = ensure_scifact_real_subset_cache()
    append_measurements_for_dataset(
        measurements,
        "SciFact",
        load_profile_snapshot(
            "scifact_real_subset_imputed_selection_dim128_kernel",
            Path(".cache/kayak/scifact_real_subset_imputed_selection_dim128_kernel"),
            scifact_cache.stored_index,
        ),
        scifact_cache.stored_task.task,
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    append_measurements_for_dataset(
        measurements,
        "FIQA",
        load_profile_snapshot(
            "fiqa_real_subset_imputed_selection_dim128_kernel",
            Path(".cache/kayak/fiqa_real_subset_imputed_selection_dim128_kernel"),
            fiqa_cache.stored_index,
        ),
        fiqa_cache.stored_task.task,
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    append_measurements_for_dataset(
        measurements,
        "LIMIT-small",
        load_profile_snapshot(
            "limit_small_real_subset_imputed_selection_dim128_kernel",
            Path(".cache/kayak/limit_small_real_subset_imputed_selection_dim128_kernel"),
            limit_small_cache.stored_index,
        ),
        limit_small_cache.stored_task.task,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root / "profile_imputed_selection_dim128_nested_kernel.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
