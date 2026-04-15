import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    ExactScoringConfig,
    JudgedTask,
    NamespaceId,
    SearchPlan,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    best_effort_faithfulness_policy,
    candidate_generation_for_plan,
    centroid_postings_imputed_flat_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    stage2_result_for_plan,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.planning.candidate_set import CandidateSet


comptime CENTROID_HEAD_POSTING_CAP = 16


struct PartitionSweepMeasurement(Copyable):
    var benchmark_name: String
    var slice_name: String
    var backend_name: String
    var candidate_k: Int
    var document_count: Int
    var query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var work_items: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var benchmark_name: String,
        var slice_name: String,
        var backend_name: String,
        candidate_k: Int,
        document_count: Int,
        query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        work_items: Int,
        mean_seconds: Float64,
    ):
        self.benchmark_name = benchmark_name^
        self.slice_name = slice_name^
        self.backend_name = backend_name^
        self.candidate_k = candidate_k
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.work_items = work_items
        self.mean_seconds = mean_seconds


def append_tsv_line(mut out: String, read measurement: PartitionSweepMeasurement):
    out += measurement.benchmark_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.backend_name
    out += "\t"
    out += String(measurement.candidate_k)
    out += "\t"
    out += String(measurement.document_count)
    out += "\t"
    out += String(measurement.query_vector_count)
    out += "\t"
    out += String(measurement.nominal_document_vector_count)
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.work_items)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\t"
    out += String(1.0 / measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[PartitionSweepMeasurement]
) raises:
    var lines = String()
    lines += "benchmark_name\tslice_name\tbackend_name\tcandidate_k\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\twork_items\tmean_seconds\tthroughput_per_second\n"

    for measurement in measurements:
        append_tsv_line(lines, measurement)

    path.write_text(lines)


def stage1_candidate_k(task: JudgedTask, document_count: Int) -> Int:
    var candidate_k = task.k * 4
    if candidate_k > document_count:
        return document_count

    return candidate_k


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


def precompute_candidate_sets(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> List[CandidateSet]:
    var candidate_sets = List[CandidateSet]()

    for judged_query in task.queries:
        candidate_sets.append(
            candidate_generation_for_plan(
                backend,
                judged_query.query,
                snapshot,
                plan,
            )
        )

    return candidate_sets^


def default_backend() -> ExactCpuBackend:
    return ExactCpuBackend()


def serial_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def conservative_parallel_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_work_item_oversubscription = False
    return ExactCpuBackend(config^)


def fixed_work_item_backend(work_items: Int) -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.parallel_work_item_count_override = work_items
    return ExactCpuBackend(config^)


def benchmark_stage2_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read candidate_sets: List[CandidateSet],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            stage2_result_for_plan(
                backend,
                task.queries[query_index].query,
                "",
                snapshot,
                candidate_sets[query_index],
                plan,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def append_measurement(
    mut measurements: List[PartitionSweepMeasurement],
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    backend_name: String,
    work_items: Int,
    mean_seconds: Float64,
):
    measurements.append(
        PartitionSweepMeasurement(
            "centroid_postings_imputed_flat_stage2_result_for_plan",
            task.slice_name.copy(),
            backend_name,
            stage1_candidate_k(task, snapshot.snapshot.stats.document_count),
            snapshot.snapshot.stats.document_count,
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            work_items,
            mean_seconds,
        )
    )


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_centroid_stage2_partition_sweep",
        Path(".cache/kayak/browsecomp_plus_gold_centroid_stage2_partition_sweep"),
        cache.stored_index,
    )
    var candidate_k = stage1_candidate_k(task, snapshot.snapshot.stats.document_count)
    var plan = centroid_postings_imputed_flat_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    var candidate_sets = precompute_candidate_sets(
        default_backend(),
        task,
        snapshot,
        plan,
    )
    var measurements = List[PartitionSweepMeasurement]()

    print(
        "dataset= BrowseComp-Plus Gold  slice= ",
        task.slice_name,
        " docs= ",
        snapshot.snapshot.stats.document_count,
        " query_vectors= ",
        task.nominal_query_vector_count,
        " doc_vectors= ",
        task.nominal_document_vector_count,
        " candidate_k= ",
        candidate_k,
    )
    print("")

    print("== stage2_result_for_plan backend=default ==")
    append_measurement(
        measurements,
        task,
        snapshot,
        "default",
        0,
        benchmark_stage2_mean_seconds(
            default_backend(),
            task,
            snapshot,
            plan,
            candidate_sets,
        ),
    )

    print("== stage2_result_for_plan backend=serial ==")
    append_measurement(
        measurements,
        task,
        snapshot,
        "serial",
        1,
        benchmark_stage2_mean_seconds(
            serial_backend(),
            task,
            snapshot,
            plan,
            candidate_sets,
        ),
    )

    print("== stage2_result_for_plan backend=parallel_oversubscription_disabled ==")
    append_measurement(
        measurements,
        task,
        snapshot,
        "parallel_oversubscription_disabled",
        0,
        benchmark_stage2_mean_seconds(
            conservative_parallel_backend(),
            task,
            snapshot,
            plan,
            candidate_sets,
        ),
    )

    for work_items in [2, 3, 4, 5, 6, 8]:
        print("== stage2_result_for_plan work_items=", work_items, " ==")
        append_measurement(
            measurements,
            task,
            snapshot,
            "work_items=" + String(work_items),
            work_items,
            benchmark_stage2_mean_seconds(
                fixed_work_item_backend(work_items),
                task,
                snapshot,
                plan,
                candidate_sets,
            ),
        )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root / "profile_centroid_stage2_partition_sweep_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
