import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    JudgedTask,
    MutableCentroidCandidateGenerationWorkspace,
    NamespaceId,
    SearchPlan,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    best_effort_faithfulness_policy,
    candidate_generation_for_plan,
    candidate_generation_for_plan_with_workspace,
    centroid_postings_flat_search_plan,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_search_plan,
    ensure_one_segment_collection_mirror,
    ensure_scifact_real_subset_cache,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot


comptime CENTROID_HEAD_POSTING_CAP = 16


struct WorkspaceReuseMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var candidate_generator_kind: String
    var mode: String
    var candidate_k: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var candidate_generator_kind: String,
        var mode: String,
        candidate_k: Int,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.candidate_generator_kind = candidate_generator_kind^
        self.mode = mode^
        self.candidate_k = candidate_k
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: WorkspaceReuseMeasurement
):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.candidate_generator_kind
    out += "\t"
    out += measurement.mode
    out += "\t"
    out += String(measurement.candidate_k)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[WorkspaceReuseMeasurement]
) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tcandidate_generator_kind\tmode\tcandidate_k\tmean_seconds\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

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


def benchmark_candidate_generation_mean_seconds_plain(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            candidate_generation_for_plan(
                backend,
                task.queries[query_index].query,
                snapshot,
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


def benchmark_candidate_generation_mean_seconds_reused(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> Float64:
    var query_index = 0
    var workspace = MutableCentroidCandidateGenerationWorkspace()

    def score_once() capturing raises:
        bench_compiler.keep(
            candidate_generation_for_plan_with_workspace(
                backend,
                task.queries[query_index].query,
                snapshot,
                plan,
                workspace,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def append_plan_measurements(
    mut measurements: List[WorkspaceReuseMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    generator_kind: String,
    candidate_k: Int,
) raises:
    var backend = ExactCpuBackend()

    print("== ", generator_kind, " plain ==")
    measurements.append(
        WorkspaceReuseMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            generator_kind.copy(),
            "fresh_workspace",
            candidate_k,
            benchmark_candidate_generation_mean_seconds_plain(
                backend,
                task,
                snapshot,
                plan,
            ),
        )
    )

    print("== ", generator_kind, " reused ==")
    measurements.append(
        WorkspaceReuseMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            generator_kind.copy(),
            "reused_workspace",
            candidate_k,
            benchmark_candidate_generation_mean_seconds_reused(
                backend,
                task,
                snapshot,
                plan,
            ),
        )
    )


def main() raises:
    var measurements = List[WorkspaceReuseMeasurement]()
    var scifact_cache = ensure_scifact_real_subset_cache()
    var snapshot = load_profile_snapshot(
        "scifact_real_subset_workspace_profile",
        Path(".cache/kayak/scifact_real_subset_workspace_profile"),
        scifact_cache.stored_index,
    )
    var candidate_k = stage1_candidate_k(
        scifact_cache.stored_task.task,
        snapshot.snapshot.stats.document_count,
    )

    print(
        "dataset=",
        "SciFact",
        " slice=",
        scifact_cache.stored_task.task.slice_name,
        " candidate_k=",
        candidate_k,
    )
    print("")

    append_plan_measurements(
        measurements,
        "SciFact",
        scifact_cache.stored_task.task,
        snapshot,
        centroid_postings_search_plan(
            scifact_cache.stored_task.task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        "centroid_postings",
        candidate_k,
    )
    append_plan_measurements(
        measurements,
        "SciFact",
        scifact_cache.stored_task.task,
        snapshot,
        centroid_postings_flat_search_plan(
            scifact_cache.stored_task.task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        "centroid_postings_flat",
        candidate_k,
    )
    append_plan_measurements(
        measurements,
        "SciFact",
        scifact_cache.stored_task.task,
        snapshot,
        centroid_postings_imputed_search_plan(
            scifact_cache.stored_task.task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        "centroid_postings_imputed",
        candidate_k,
    )
    append_plan_measurements(
        measurements,
        "SciFact",
        scifact_cache.stored_task.task,
        snapshot,
        centroid_postings_imputed_flat_search_plan(
            scifact_cache.stored_task.task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        "centroid_postings_imputed_flat",
        candidate_k,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_centroid_workspace_reuse.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
