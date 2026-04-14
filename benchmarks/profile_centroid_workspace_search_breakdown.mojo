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
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    search_collection_for_plan,
    search_collection_for_plan_with_workspace,
    stage2_result_for_plan,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.collection_hit import CollectionHit


comptime CENTROID_HEAD_POSTING_CAP = 16


struct WorkspaceSearchBreakdownMeasurement(Copyable):
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
    mut out: String, read measurement: WorkspaceSearchBreakdownMeasurement
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
    path: Path, measurements: List[WorkspaceSearchBreakdownMeasurement]
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


def hits_match(
    read lhs: List[CollectionHit], read rhs: List[CollectionHit]
) -> Bool:
    if len(lhs) != len(rhs):
        return False

    for index in range(len(lhs)):
        if lhs[index].segment_id != rhs[index].segment_id:
            return False
        if lhs[index].doc_id != rhs[index].doc_id:
            return False

    return True


def candidate_sets_match(
    read lhs: CandidateSet, read rhs: CandidateSet
) -> Bool:
    if lhs.generator_kind != rhs.generator_kind:
        return False

    return hits_match(lhs.hits, rhs.hits)


def verify_plan_results_match(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> List[CandidateSet]:
    var workspace = MutableCentroidCandidateGenerationWorkspace()
    var fresh_candidate_sets = List[CandidateSet]()

    for judged_query in task.queries:
        var fresh_candidates = candidate_generation_for_plan(
            backend,
            judged_query.query,
            snapshot,
            plan,
        )
        var reused_candidates = candidate_generation_for_plan_with_workspace(
            backend,
            judged_query.query,
            snapshot,
            plan,
            workspace,
        )
        if not candidate_sets_match(fresh_candidates, reused_candidates):
            raise Error(
                "workspace search breakdown benchmark observed candidate-set mismatch for "
                + plan.candidate_generator.kind
            )

        var fresh_hits = search_collection_for_plan(
            backend,
            judged_query.query,
            snapshot,
            plan,
        )
        var reused_hits = search_collection_for_plan_with_workspace(
            backend,
            judged_query.query,
            snapshot,
            plan,
            workspace,
        )
        if not hits_match(fresh_hits, reused_hits):
            raise Error(
                "workspace search breakdown benchmark observed final-hit mismatch for "
                + plan.candidate_generator.kind
            )

        fresh_candidate_sets.append(fresh_candidates^)

    return fresh_candidate_sets^


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


def benchmark_search_mean_seconds_plain(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            search_collection_for_plan(
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


def benchmark_search_mean_seconds_reused(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> Float64:
    var query_index = 0
    var workspace = MutableCentroidCandidateGenerationWorkspace()

    def score_once() capturing raises:
        bench_compiler.keep(
            search_collection_for_plan_with_workspace(
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


def append_measurement(
    mut measurements: List[WorkspaceSearchBreakdownMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    generator_kind: String,
    benchmark_kind: String,
    candidate_k: Int,
    mean_seconds: Float64,
):
    measurements.append(
        WorkspaceSearchBreakdownMeasurement(
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
    mut measurements: List[WorkspaceSearchBreakdownMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    candidate_k: Int,
) raises:
    var backend = ExactCpuBackend()
    var candidate_sets = verify_plan_results_match(backend, task, snapshot, plan)

    print("== ", plan.candidate_generator.kind, " stage1 plain ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "stage1_fresh_workspace",
        candidate_k,
        benchmark_candidate_generation_mean_seconds_plain(
            backend,
            task,
            snapshot,
            plan,
        ),
    )

    print("== ", plan.candidate_generator.kind, " stage1 reused ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "stage1_reused_workspace",
        candidate_k,
        benchmark_candidate_generation_mean_seconds_reused(
            backend,
            task,
            snapshot,
            plan,
        ),
    )

    print("== ", plan.candidate_generator.kind, " stage2_from_candidates ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "stage2_from_candidates",
        candidate_k,
        benchmark_stage2_mean_seconds(
            backend,
            task,
            snapshot,
            plan,
            candidate_sets,
        ),
    )

    print("== ", plan.candidate_generator.kind, " search plain ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "search_fresh_workspace",
        candidate_k,
        benchmark_search_mean_seconds_plain(
            backend,
            task,
            snapshot,
            plan,
        ),
    )

    print("== ", plan.candidate_generator.kind, " search reused ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "search_reused_workspace",
        candidate_k,
        benchmark_search_mean_seconds_reused(
            backend,
            task,
            snapshot,
            plan,
        ),
    )


def main() raises:
    var measurements = List[WorkspaceSearchBreakdownMeasurement]()
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_workspace_search_breakdown",
        Path(".cache/kayak/browsecomp_plus_gold_workspace_search_breakdown"),
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
    var output_path = output_root / "profile_centroid_workspace_search_breakdown.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
