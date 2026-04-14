import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    MutableCentroidCandidateGenerationWorkspace,
    NamespaceId,
    SnapshotId,
    TenantId,
    candidate_generation_for_plan,
    candidate_generation_for_plan_with_workspace,
    centroid_postings_flat_search_plan,
    centroid_postings_search_plan,
    ensure_one_segment_collection_mirror,
    loaded_segment_stored_centroid_postings_index,
    load_resolved_collection_snapshot,
)
from kayak.benchmarks import (
    SingleCoreScaleProfile,
    default_single_core_scale_profiles,
    make_single_core_scale_fixture,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.eval import JudgedTask
from kayak.planning import SearchPlan, best_effort_faithfulness_policy
from kayak.storage import StoredPackedIndex


comptime CENTROID_HEAD_POSTING_CAP = 16


struct WorkspaceScaleMeasurement(Copyable):
    var slice_name: String
    var candidate_generator_kind: String
    var mode: String
    var document_count: Int
    var query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var candidate_k: Int
    var centroid_count: Int
    var total_posting_count: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var slice_name: String,
        var candidate_generator_kind: String,
        var mode: String,
        document_count: Int,
        query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        candidate_k: Int,
        centroid_count: Int,
        total_posting_count: Int,
        mean_seconds: Float64,
    ):
        self.slice_name = slice_name^
        self.candidate_generator_kind = candidate_generator_kind^
        self.mode = mode^
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.candidate_k = candidate_k
        self.centroid_count = centroid_count
        self.total_posting_count = total_posting_count
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: WorkspaceScaleMeasurement
):
    out += measurement.slice_name
    out += "\t"
    out += measurement.candidate_generator_kind
    out += "\t"
    out += measurement.mode
    out += "\t"
    out += String(measurement.document_count)
    out += "\t"
    out += String(measurement.query_vector_count)
    out += "\t"
    out += String(measurement.nominal_document_vector_count)
    out += "\t"
    out += String(measurement.vector_dim)
    out += "\t"
    out += String(measurement.candidate_k)
    out += "\t"
    out += String(measurement.centroid_count)
    out += "\t"
    out += String(measurement.total_posting_count)
    out += "\t"
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[WorkspaceScaleMeasurement]
) raises:
    var lines = String()
    lines += "slice_name\tcandidate_generator_kind\tmode\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tcandidate_k\tcentroid_count\ttotal_posting_count\tmean_seconds\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def extended_scale_profiles() raises -> List[SingleCoreScaleProfile]:
    var profiles = default_single_core_scale_profiles()
    profiles.append(
        SingleCoreScaleProfile(
            "synthetic_scale",
            "docs_16384",
            "Single-core sweep that increases only corpus size to 16384 documents.",
            16384,
            8,
            4,
            8,
            64,
            2,
            16,
            16,
        )
    )
    return profiles^


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


def append_measurements_for_plan(
    mut measurements: List[WorkspaceScaleMeasurement],
    read profile: SingleCoreScaleProfile,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    generator_kind: String,
    read plan: SearchPlan,
    candidate_k: Int,
) raises:
    var backend = ExactCpuBackend()
    var centroid_index = loaded_segment_stored_centroid_postings_index(
        snapshot.segments[0]
    ).index.copy()

    print(
        "== ",
        generator_kind,
        " plain docs=",
        snapshot.snapshot.stats.document_count,
        " centroids=",
        centroid_index.centroid_count,
        " postings=",
        centroid_index.total_posting_count,
        " ==",
    )
    measurements.append(
        WorkspaceScaleMeasurement(
            profile.slice_name.copy(),
            generator_kind.copy(),
            "fresh_workspace",
            snapshot.snapshot.stats.document_count,
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            candidate_k,
            centroid_index.centroid_count,
            centroid_index.total_posting_count,
            benchmark_candidate_generation_mean_seconds_plain(
                backend,
                task,
                snapshot,
                plan,
            ),
        )
    )

    print(
        "== ",
        generator_kind,
        " reused docs=",
        snapshot.snapshot.stats.document_count,
        " centroids=",
        centroid_index.centroid_count,
        " postings=",
        centroid_index.total_posting_count,
        " ==",
    )
    measurements.append(
        WorkspaceScaleMeasurement(
            profile.slice_name.copy(),
            generator_kind.copy(),
            "reused_workspace",
            snapshot.snapshot.stats.document_count,
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            candidate_k,
            centroid_index.centroid_count,
            centroid_index.total_posting_count,
            benchmark_candidate_generation_mean_seconds_reused(
                backend,
                task,
                snapshot,
                plan,
            ),
        )
    )


def append_measurements_for_profile(
    mut measurements: List[WorkspaceScaleMeasurement],
    read profile: SingleCoreScaleProfile,
) raises:
    var fixture = make_single_core_scale_fixture(profile)
    var snapshot = load_profile_snapshot(
        "centroid-workspace-scale-" + profile.slice_name,
        Path(".cache/kayak/centroid_workspace_scale_" + profile.slice_name),
        fixture.stored_index,
    )
    var candidate_k = profile.candidate_k
    if candidate_k > snapshot.snapshot.stats.document_count:
        candidate_k = snapshot.snapshot.stats.document_count

    print(
        "slice=",
        profile.slice_name,
        " docs=",
        snapshot.snapshot.stats.document_count,
        " query_vectors=",
        fixture.stored_task.task.nominal_query_vector_count,
        " doc_vectors=",
        fixture.stored_task.task.nominal_document_vector_count,
        " candidate_k=",
        candidate_k,
    )
    print("")

    append_measurements_for_plan(
        measurements,
        profile,
        fixture.stored_task.task,
        snapshot,
        "centroid_postings",
        centroid_postings_search_plan(
            profile.final_k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        candidate_k,
    )
    append_measurements_for_plan(
        measurements,
        profile,
        fixture.stored_task.task,
        snapshot,
        "centroid_postings_flat",
        centroid_postings_flat_search_plan(
            profile.final_k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        candidate_k,
    )


def main() raises:
    var measurements = List[WorkspaceScaleMeasurement]()

    for profile in extended_scale_profiles():
        append_measurements_for_profile(measurements, profile)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_centroid_workspace_scale.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
