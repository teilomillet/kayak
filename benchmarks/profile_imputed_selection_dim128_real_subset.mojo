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
from kayak.contracts import FlatQueryDim128, build_flat_query_dim128
from kayak.planning.centroid_postings_imputed_flat_stage import (
    centroid_selection_for_flat_query_token_dim128,
)
from kayak.planning.centroid_postings_imputed_stage import (
    centroid_selection_for_query_token,
)


comptime CENTROID_HEAD_POSTING_CAP = 16


struct ImputedSelectionMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var primitive_kind: String
    var vector_dim: Int
    var centroid_count: Int
    var final_k: Int
    var query_count: Int
    var total_query_vectors: Int
    var max_query_vectors: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var primitive_kind: String,
        vector_dim: Int,
        centroid_count: Int,
        final_k: Int,
        query_count: Int,
        total_query_vectors: Int,
        max_query_vectors: Int,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.primitive_kind = primitive_kind^
        self.vector_dim = vector_dim
        self.centroid_count = centroid_count
        self.final_k = final_k
        self.query_count = query_count
        self.total_query_vectors = total_query_vectors
        self.max_query_vectors = max_query_vectors
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: ImputedSelectionMeasurement
):
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
    out += String(measurement.final_k)
    out += "\t"
    out += String(measurement.query_count)
    out += "\t"
    out += String(measurement.total_query_vectors)
    out += "\t"
    out += String(measurement.max_query_vectors)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[ImputedSelectionMeasurement]
) raises:
    var lines = String()
    lines += (
        "dataset_name\tslice_name\tprimitive_kind\tvector_dim\tcentroid_count"
        "\tfinal_k\tquery_count\ttotal_query_vectors\tmax_query_vectors"
        "\tmean_seconds\n"
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


def max_query_vectors(read task: JudgedTask) -> Int:
    var max_count = 0
    for judged_query in task.queries:
        if judged_query.query.vector_count > max_count:
            max_count = judged_query.query.vector_count
    return max_count


def imputed_nested_selections_for_query(
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


def imputed_flat_selections_for_query(
    read snapshot: ResolvedCollectionSnapshot,
    read flat_queries: List[FlatQueryDim128],
    read task: JudgedTask,
    query_index: Int,
) raises -> List[ScoredCentroidSelection]:
    var selections = List[ScoredCentroidSelection]()
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    for token_index in range(flat_queries[query_index].vector_count):
        selections.append(
            centroid_selection_for_flat_query_token_dim128(
                flat_queries[query_index],
                token_index,
                index,
                task.k,
            )
        )

    return selections^


def append_measurements_for_dataset(
    mut measurements: List[ImputedSelectionMeasurement],
    dataset_name: String,
    read snapshot: ResolvedCollectionSnapshot,
    read task: JudgedTask,
) raises:
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()
    if index.vector_dim != 128:
        raise Error("dim128 imputed selection benchmark requires vector_dim == 128")
    if index.centroid_count > 128:
        raise Error(
            "dim128 imputed selection benchmark requires centroid_count <= 128"
        )

    var flat_queries = List[FlatQueryDim128]()
    for judged_query in task.queries:
        flat_queries.append(build_flat_query_dim128(judged_query.query))

    var total_vectors = total_query_vectors(task)
    var max_vectors = max_query_vectors(task)

    print(
        "dataset= ",
        dataset_name,
        " slice= ",
        task.slice_name,
        " vector_dim= ",
        index.vector_dim,
        " centroid_count= ",
        index.centroid_count,
        " final_k= ",
        task.k,
        " query_count= ",
        len(task.queries),
        " total_query_vectors= ",
        total_vectors,
        " max_query_vectors= ",
        max_vectors,
    )
    print("")

    var query_index = 0
    print("== imputed_selection_nested ==")

    def nested_once() capturing raises:
        bench_compiler.keep(
            imputed_nested_selections_for_query(snapshot, task, query_index)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[nested_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    report.print()
    print("")
    measurements.append(
        ImputedSelectionMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "imputed_selection_nested",
            index.vector_dim,
            index.centroid_count,
            task.k,
            len(task.queries),
            total_vectors,
            max_vectors,
            report.mean(),
        )
    )

    query_index = 0
    print("== imputed_selection_flat ==")

    def flat_once() capturing raises:
        bench_compiler.keep(
            imputed_flat_selections_for_query(snapshot, flat_queries, task, query_index)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    report = benchmark.run[flat_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    report.print()
    print("")
    measurements.append(
        ImputedSelectionMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            "imputed_selection_flat",
            index.vector_dim,
            index.centroid_count,
            task.k,
            len(task.queries),
            total_vectors,
            max_vectors,
            report.mean(),
        )
    )


def main() raises:
    var measurements = List[ImputedSelectionMeasurement]()

    var scifact_cache = ensure_scifact_real_subset_cache()
    append_measurements_for_dataset(
        measurements,
        "SciFact",
        load_profile_snapshot(
            "scifact_real_subset_imputed_selection_dim128",
            Path(".cache/kayak/scifact_real_subset_imputed_selection_dim128"),
            scifact_cache.stored_index,
        ),
        scifact_cache.stored_task.task,
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    append_measurements_for_dataset(
        measurements,
        "FIQA",
        load_profile_snapshot(
            "fiqa_real_subset_imputed_selection_dim128",
            Path(".cache/kayak/fiqa_real_subset_imputed_selection_dim128"),
            fiqa_cache.stored_index,
        ),
        fiqa_cache.stored_task.task,
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    append_measurements_for_dataset(
        measurements,
        "LIMIT-small",
        load_profile_snapshot(
            "limit_small_real_subset_imputed_selection_dim128",
            Path(".cache/kayak/limit_small_real_subset_imputed_selection_dim128"),
            limit_small_cache.stored_index,
        ),
        limit_small_cache.stored_task.task,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_imputed_selection_dim128_real_subset.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
