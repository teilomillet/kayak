import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    CollectionHit,
    EncodedQuery,
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
    ResolvedCandidateWindow,
    exact_rerank_candidates_for_plan,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)
from kayak.planning.topk import insert_descending_collection_hit


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


def full_score_reference_rerank(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_set: CandidateSet,
    final_k: Int,
) raises -> List[CollectionHit]:
    var resolved_window = resolve_candidate_window(snapshot, candidate_set.hits)
    var scores = score_resolved_candidate_window_for_cpu(
        backend,
        query,
        snapshot,
        resolved_window,
    )
    return assemble_topk_hits(
        candidate_set,
        resolved_window,
        scores,
        final_k,
    )


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


def benchmark_exact_rerank_full_score_reference_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_sets: List[CandidateSet],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            full_score_reference_rerank(
                backend,
                task.queries[query_index].query,
                snapshot,
                candidate_sets[query_index],
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
    read backend: ExactCpuBackend,
) raises:
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

    print("== ", plan.candidate_generator.kind, " resolve_candidate_window ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "resolve_candidate_window",
        candidate_k,
        benchmark_resolve_mean_seconds(task, snapshot, candidate_sets),
    )
    print(
        "== ",
        plan.candidate_generator.kind,
        " score_resolved_candidate_window_for_cpu ==",
    )
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_score_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )
    print("== ", plan.candidate_generator.kind, " assemble_topk_hits ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "assemble_topk_hits",
        candidate_k,
        benchmark_assemble_mean_seconds(
            task,
            candidate_sets,
            resolved_windows,
            precomputed_scores,
        ),
    )
    print("== ", plan.candidate_generator.kind, " exact_rerank_full_score_reference ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "exact_rerank_full_score_reference",
        candidate_k,
        benchmark_exact_rerank_full_score_reference_mean_seconds(
            backend,
            task,
            snapshot,
            candidate_sets,
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
    var backend = ExactCpuBackend()
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var candidate_k = stage1_candidate_k(
        task,
        cache.stored_index.index.document_count,
    )
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_centroid_exact_stage_breakdown",
        Path(".cache/kayak/browsecomp_plus_gold_centroid_exact_stage_breakdown"),
        cache.stored_index,
    )

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

    var measurements = List[ExactStageBreakdownMeasurement]()
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
        backend,
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
        backend,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root / "profile_centroid_exact_stage_breakdown_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
