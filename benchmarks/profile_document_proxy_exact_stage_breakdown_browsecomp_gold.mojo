import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    CollectionHit,
    ExactCpuBackend,
    NamespaceId,
    SearchPlan,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    best_effort_faithfulness_policy,
    candidate_generation_for_plan,
    document_proxy_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.eval import JudgedTask
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    ResolvedCandidateWindow,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)
from kayak.planning.topk import insert_descending_collection_hit


struct ExactStageBreakdownMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
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
    lines += "dataset_name\tslice_name\tbenchmark_kind\tcandidate_k\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tmean_seconds\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def load_profile_snapshot(
    collection_name: String,
    collection_root: Path,
    read stored_index: StoredPackedIndex,
    document_proxy_vector_budget: Int,
) raises -> ResolvedCollectionSnapshot:
    var root = ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId(collection_name),
        TenantId("public"),
        NamespaceId("benchmark"),
        SnapshotId("snapshot-0001"),
        1,
        stored_index,
        document_proxy_vector_budget,
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


def precompute_resolved_windows(
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_sets: List[CandidateSet],
) raises -> List[ResolvedCandidateWindow]:
    var resolved_windows = List[ResolvedCandidateWindow]()

    for candidate_set in candidate_sets:
        resolved_windows.append(
            resolve_candidate_window(snapshot, candidate_set.hits)
        )

    return resolved_windows^


def benchmark_resolve_mean_seconds(
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_sets: List[CandidateSet],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            resolve_candidate_window(snapshot, candidate_sets[query_index].hits)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_score_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            score_resolved_candidate_window_for_cpu(
                backend,
                task.queries[query_index].query,
                snapshot,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def assemble_topk_hits(
    read candidate_set: CandidateSet,
    read resolved_window: ResolvedCandidateWindow,
    read scores: List[Float32],
    final_k: Int,
) raises -> List[CollectionHit]:
    var final_hits = List[CollectionHit]()

    for document_index in range(len(scores)):
        insert_descending_collection_hit(
            final_hits,
            CollectionHit(
                resolved_window.documents[document_index].segment_id.copy(),
                candidate_set.hits[document_index].doc_id.copy(),
                scores[document_index],
            ),
            final_k,
        )

    return final_hits^


def benchmark_assemble_mean_seconds(
    read task: JudgedTask,
    read candidate_sets: List[CandidateSet],
    read resolved_windows: List[ResolvedCandidateWindow],
    read precomputed_scores: List[List[Float32]],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            assemble_topk_hits(
                candidate_sets[query_index],
                resolved_windows[query_index],
                precomputed_scores[query_index],
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


def precompute_scores(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> List[List[Float32]]:
    var all_scores = List[List[Float32]]()

    for query_index in range(len(task.queries)):
        all_scores.append(
            score_resolved_candidate_window_for_cpu(
                backend,
                task.queries[query_index].query,
                snapshot,
                resolved_windows[query_index],
            )
        )

    return all_scores^


def append_measurement(
    mut measurements: List[ExactStageBreakdownMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    benchmark_kind: String,
    candidate_k: Int,
    mean_seconds: Float64,
):
    measurements.append(
        ExactStageBreakdownMeasurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            benchmark_kind.copy(),
            candidate_k,
            snapshot.snapshot.stats.document_count,
            task.nominal_query_vector_count,
            task.nominal_document_vector_count,
            task.vector_dim,
            mean_seconds,
        )
    )


def main() raises:
    var backend = ExactCpuBackend()
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var candidate_k = task.k * 4
    var proxy_budget = task.nominal_document_vector_count
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_document_proxy_exact_stage_breakdown",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_exact_stage_breakdown"),
        cache.stored_index,
        proxy_budget,
    )
    var plan = document_proxy_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    var candidate_sets = precompute_candidate_sets(
        backend,
        task,
        snapshot,
        plan,
    )
    var resolved_windows = precompute_resolved_windows(snapshot, candidate_sets)
    var precomputed_scores = precompute_scores(
        backend,
        task,
        snapshot,
        resolved_windows,
    )
    var measurements = List[ExactStageBreakdownMeasurement]()

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

    print("== resolve_candidate_window ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "resolve_candidate_window",
        candidate_k,
        benchmark_resolve_mean_seconds(task, snapshot, candidate_sets),
    )
    print("== score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_score_mean_seconds(backend, task, snapshot, resolved_windows),
    )
    print("== assemble_topk_hits ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "assemble_topk_hits",
        candidate_k,
        benchmark_assemble_mean_seconds(
            task,
            candidate_sets,
            resolved_windows,
            precomputed_scores,
        ),
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root / "profile_document_proxy_exact_stage_breakdown_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
