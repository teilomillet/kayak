import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    JudgedTask,
    NamespaceId,
    SearchPlan,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    best_effort_faithfulness_policy,
    candidate_generation_for_plan,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    MaterializedCandidateIndex,
    exact_rerank_candidates_for_plan,
    materialize_candidate_index,
)


comptime CENTROID_HEAD_POSTING_CAP = 16


struct ExactStageBreakdownMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var candidate_generator_kind: String
    var benchmark_kind: String
    var candidate_k: Int
    var document_count: Int
    var query_vector_count: Int
    var nominal_document_vector_count: Int
    var vector_dim: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var candidate_generator_kind: String,
        var benchmark_kind: String,
        candidate_k: Int,
        document_count: Int,
        query_vector_count: Int,
        nominal_document_vector_count: Int,
        vector_dim: Int,
        mean_seconds: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.candidate_generator_kind = candidate_generator_kind^
        self.benchmark_kind = benchmark_kind^
        self.candidate_k = candidate_k
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: ExactStageBreakdownMeasurement
):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.candidate_generator_kind
    out += "\t"
    out += measurement.benchmark_kind
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
    out += String(measurement.mean_seconds)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[ExactStageBreakdownMeasurement]
) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tcandidate_generator_kind\tbenchmark_kind\tcandidate_k\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tmean_seconds\n"

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


def precompute_materialized_indices(
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_sets: List[CandidateSet],
) raises -> List[MaterializedCandidateIndex]:
    var materialized = List[MaterializedCandidateIndex]()

    for candidate_set in candidate_sets:
        materialized.append(materialize_candidate_index(snapshot, candidate_set.hits))

    return materialized^


def benchmark_materialize_mean_seconds(
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_sets: List[CandidateSet],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            materialize_candidate_index(
                snapshot,
                candidate_sets[query_index].hits,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_score_materialized_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read materialized_indices: List[MaterializedCandidateIndex],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            backend.score_all(
                task.queries[query_index].query,
                materialized_indices[query_index].index,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_exact_rerank_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_sets: List[CandidateSet],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            exact_rerank_candidates_for_plan(
                backend,
                task.queries[query_index].query,
                snapshot,
                candidate_sets[query_index].hits,
                task.k,
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
    mut measurements: List[ExactStageBreakdownMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    generator_kind: String,
    benchmark_kind: String,
    candidate_k: Int,
    mean_seconds: Float64,
):
    measurements.append(
        ExactStageBreakdownMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            generator_kind.copy(),
            benchmark_kind.copy(),
            candidate_k,
            snapshot.snapshot.stats.document_count,
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            mean_seconds,
        )
    )


def append_plan_measurements(
    mut measurements: List[ExactStageBreakdownMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    candidate_k: Int,
) raises:
    var backend = ExactCpuBackend()
    var candidate_sets = precompute_candidate_sets(backend, task, snapshot, plan)
    var materialized_indices = precompute_materialized_indices(
        snapshot,
        candidate_sets,
    )

    print("== ", plan.candidate_generator.kind, " materialize_candidate_index ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "materialize_candidate_index",
        candidate_k,
        benchmark_materialize_mean_seconds(task, snapshot, candidate_sets),
    )

    print("== ", plan.candidate_generator.kind, " score_materialized_index ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "score_materialized_index",
        candidate_k,
        benchmark_score_materialized_mean_seconds(
            backend,
            task,
            materialized_indices,
        ),
    )

    print("== ", plan.candidate_generator.kind, " exact_rerank_candidates_for_plan ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "exact_rerank_candidates_for_plan",
        candidate_k,
        benchmark_exact_rerank_mean_seconds(
            backend,
            task,
            snapshot,
            candidate_sets,
        ),
    )


def main() raises:
    var measurements = List[ExactStageBreakdownMeasurement]()
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_exact_stage_breakdown",
        Path(".cache/kayak/browsecomp_plus_gold_exact_stage_breakdown"),
        cache.stored_index,
    )
    var task = cache.stored_task.task.copy()
    var candidate_k = stage1_candidate_k(task, snapshot.snapshot.stats.document_count)

    print(
        "dataset=",
        "BrowseComp-Plus Gold",
        " slice=",
        task.slice_name,
        " docs=",
        snapshot.snapshot.stats.document_count,
        " query_vectors=",
        task.nominal_query_vector_count,
        " doc_vectors=",
        task.nominal_document_vector_count,
        " candidate_k=",
        candidate_k,
    )
    print("")

    append_plan_measurements(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        centroid_postings_imputed_search_plan(
            task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        candidate_k,
    )
    append_plan_measurements(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        centroid_postings_imputed_flat_search_plan(
            task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        ),
        candidate_k,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_exact_stage_materialization_browsecomp_gold.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
