import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    JudgedTask,
    MutableCentroidSegmentAccumulator,
    NamespaceId,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    loaded_segment_stored_centroid_postings_index,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import FlatQueryDim128, build_flat_query_dim128
from kayak.index import CentroidPostingIndex
from kayak.planning.centroid_postings_imputed_flat_stage import (
    centroid_posting_imputed_flat_score_result_for_segment_dim128_with_workspace,
    centroid_posting_imputed_flat_score_result_for_segment_with_workspace,
)


comptime CENTROID_HEAD_POSTING_CAP = 16


struct FlatQueryHoistMeasurement(Copyable):
    var benchmark_name: String
    var slice_name: String
    var path_kind: String
    var segment_repeat_count: Int
    var document_count: Int
    var query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var benchmark_name: String,
        var slice_name: String,
        var path_kind: String,
        segment_repeat_count: Int,
        document_count: Int,
        query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        mean_seconds: Float64,
    ):
        self.benchmark_name = benchmark_name^
        self.slice_name = slice_name^
        self.path_kind = path_kind^
        self.segment_repeat_count = segment_repeat_count
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.mean_seconds = mean_seconds


def append_tsv_line(mut out: String, read measurement: FlatQueryHoistMeasurement):
    out += measurement.benchmark_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.path_kind
    out += "\t"
    out += String(measurement.segment_repeat_count)
    out += "\t"
    out += String(measurement.document_count)
    out += "\t"
    out += String(measurement.query_vector_count)
    out += "\t"
    out += String(measurement.nominal_document_vector_count)
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[FlatQueryHoistMeasurement]
) raises:
    var lines = String()
    lines += "benchmark_name\tslice_name\tpath_kind\tsegment_repeat_count\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tmean_seconds\n"

    for measurement in measurements:
        append_tsv_line(lines, measurement)

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


def benchmark_generic_mean_seconds(
    read task: JudgedTask,
    read index: CentroidPostingIndex,
    segment_repeat_count: Int,
) raises -> Float64:
    var query_index = 0
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)

    def score_once() capturing raises:
        var result = centroid_posting_imputed_flat_score_result_for_segment_with_workspace(
            task.queries[query_index].query,
            index,
            task.k,
            workspace,
        )
        for _ in range(1, segment_repeat_count):
            result = centroid_posting_imputed_flat_score_result_for_segment_with_workspace(
                task.queries[query_index].query,
                index,
                task.k,
                workspace,
            )
        bench_compiler.keep(result)
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_preflattened_mean_seconds(
    read task: JudgedTask,
    read index: CentroidPostingIndex,
    segment_repeat_count: Int,
) raises -> Float64:
    var query_index = 0
    var workspace = MutableCentroidSegmentAccumulator(index.document_count)
    var flat_queries = List[FlatQueryDim128]()

    for judged_query in task.queries:
        flat_queries.append(build_flat_query_dim128(judged_query.query))

    def score_once() capturing raises:
        var result = centroid_posting_imputed_flat_score_result_for_segment_dim128_with_workspace(
            flat_queries[query_index],
            index,
            task.k,
            workspace,
        )
        for _ in range(1, segment_repeat_count):
            result = centroid_posting_imputed_flat_score_result_for_segment_dim128_with_workspace(
                flat_queries[query_index],
                index,
                task.k,
                workspace,
            )
        bench_compiler.keep(result)
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def append_measurements_for_repeat_count(
    mut measurements: List[FlatQueryHoistMeasurement],
    read task: JudgedTask,
    read index: CentroidPostingIndex,
    segment_repeat_count: Int,
) raises:
    print(
        "== imputed_flat_generic segment_repeat_count=",
        segment_repeat_count,
        " ==",
    )
    measurements.append(
        FlatQueryHoistMeasurement(
            "imputed_flat_query_hoist",
            task.slice_name.copy(),
            "generic_entrypoint",
            segment_repeat_count,
            index.document_count,
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_generic_mean_seconds(task, index, segment_repeat_count),
        )
    )

    print(
        "== imputed_flat_preflattened_dim128 segment_repeat_count=",
        segment_repeat_count,
        " ==",
    )
    measurements.append(
        FlatQueryHoistMeasurement(
            "imputed_flat_query_hoist",
            task.slice_name.copy(),
            "preflattened_dim128",
            segment_repeat_count,
            index.document_count,
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            benchmark_preflattened_mean_seconds(task, index, segment_repeat_count),
        )
    )


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_imputed_flat_query_hoist",
        Path(".cache/kayak/browsecomp_plus_gold_imputed_flat_query_hoist"),
        cache.stored_index,
    )
    var index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()
    var measurements = List[FlatQueryHoistMeasurement]()

    print(
        "dataset= BrowseComp-Plus Gold  slice= ",
        task.slice_name,
        " docs= ",
        index.document_count,
        " query_vectors= ",
        task.nominal_query_vector_count,
        " doc_vectors= ",
        task.nominal_document_vector_count,
        " vector_dim= ",
        task.vector_dim,
    )
    print("")

    for segment_repeat_count in [1, 4, 8, 16, 32]:
        append_measurements_for_repeat_count(
            measurements,
            task,
            index,
            segment_repeat_count,
        )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root / "profile_imputed_flat_query_hoist_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
