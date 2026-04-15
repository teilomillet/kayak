import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    NamespaceId,
    SearchPlan,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    best_effort_faithfulness_policy,
    build_flat_query_dim128,
    candidate_generation_for_plan,
    document_proxy_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import FlatQueryDim128
from kayak.eval import JudgedTask
from kayak.index import HybridFlatDim128Index
from kayak.numeric import ScoreScalar, VectorScalar
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    ResolvedCandidateWindow,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)
from kayak.scoring import (
    exact_scores_for_hybrid_flat_only_index_dim128,
    exact_scores_for_hybrid_flat_only_index_dim128_with_flat_query,
)


struct CandidateFlatMeasurement(Copyable):
    var benchmark_kind: String
    var slice_name: String
    var candidate_k: Int
    var query_count: Int
    var mean_candidate_document_count: Float64
    var mean_candidate_vector_count: Float64
    var mean_seconds: Float64

    def __init__(
        out self,
        var benchmark_kind: String,
        var slice_name: String,
        candidate_k: Int,
        query_count: Int,
        mean_candidate_document_count: Float64,
        mean_candidate_vector_count: Float64,
        mean_seconds: Float64,
    ):
        self.benchmark_kind = benchmark_kind^
        self.slice_name = slice_name^
        self.candidate_k = candidate_k
        self.query_count = query_count
        self.mean_candidate_document_count = mean_candidate_document_count
        self.mean_candidate_vector_count = mean_candidate_vector_count
        self.mean_seconds = mean_seconds


def write_measurements_tsv(
    path: Path, measurements: List[CandidateFlatMeasurement]
) raises:
    var lines = String()
    lines += "benchmark_kind\tslice_name\tcandidate_k\tquery_count\tmean_candidate_document_count\tmean_candidate_vector_count\tmean_seconds\tthroughput_per_second\n"

    for measurement in measurements:
        lines += measurement.benchmark_kind
        lines += "\t"
        lines += measurement.slice_name
        lines += "\t"
        lines += String(measurement.candidate_k)
        lines += "\t"
        lines += String(measurement.query_count)
        lines += "\t"
        lines += String(measurement.mean_candidate_document_count)
        lines += "\t"
        lines += String(measurement.mean_candidate_vector_count)
        lines += "\t"
        lines += String(measurement.mean_seconds)
        lines += "\t"
        lines += String(1.0 / measurement.mean_seconds)
        lines += "\n"

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


def precompute_flat_queries(read task: JudgedTask) raises -> List[FlatQueryDim128]:
    var flat_queries = List[FlatQueryDim128]()

    for judged_query in task.queries:
        flat_queries.append(build_flat_query_dim128(judged_query.query))

    return flat_queries^


def build_candidate_window_hybrid_flat_index(
    read snapshot: ResolvedCollectionSnapshot,
    read resolved: ResolvedCandidateWindow,
) raises -> HybridFlatDim128Index:
    var doc_ids = List[String]()
    var doc_offsets = List[Int]()
    var token_values = List[VectorScalar]()
    var running_vector_count = 0

    doc_offsets.append(0)

    for resolved_document in resolved.documents:
        var segment_index = resolved_document.segment_index
        var start = snapshot.segments[segment_index].stored_index.index.doc_offsets[
            resolved_document.document_index
        ]
        var stop = snapshot.segments[segment_index].stored_index.index.doc_offsets[
            resolved_document.document_index + 1
        ]

        doc_ids.append(resolved_document.doc_id.copy())

        for token_index in range(start, stop):
            for value in snapshot.segments[segment_index].stored_index.index.token_vectors[token_index]:
                token_values.append(value)

        running_vector_count += stop - start
        doc_offsets.append(running_vector_count)

    return HybridFlatDim128Index(
        doc_ids^,
        doc_offsets^,
        token_values^,
        snapshot.collection.vector_dim,
    )


def precompute_hybrid_candidate_windows(
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> List[HybridFlatDim128Index]:
    var hybrid_windows = List[HybridFlatDim128Index]()

    for resolved in resolved_windows:
        hybrid_windows.append(
            build_candidate_window_hybrid_flat_index(snapshot, resolved)
        )

    return hybrid_windows^


def require_score_lists_close(
    benchmark_kind: String,
    read expected: List[ScoreScalar],
    read observed: List[ScoreScalar],
    query_index: Int,
) raises:
    if len(expected) != len(observed):
        raise Error(
            benchmark_kind
            + " score count mismatch for query "
            + String(query_index)
        )

    for document_index in range(len(expected)):
        var difference = (
            Float64(expected[document_index]) - Float64(observed[document_index])
        )
        if difference < 0.0:
            difference = 0.0 - difference

        if difference > 1e-5:
            raise Error(
                benchmark_kind
                + " score mismatch for query "
                + String(query_index)
                + " document "
                + String(document_index)
                + " expected="
                + String(expected[document_index])
                + " observed="
                + String(observed[document_index])
                + " difference="
                + String(difference)
            )


def require_flat_candidate_window_scores_match(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
    read hybrid_windows: List[HybridFlatDim128Index],
    read flat_queries: List[FlatQueryDim128],
) raises:
    for query_index in range(len(task.queries)):
        var nested_scores = score_resolved_candidate_window_for_cpu(
            backend,
            task.queries[query_index].query,
            snapshot,
            resolved_windows[query_index],
        )
        var flat_scores = exact_scores_for_hybrid_flat_only_index_dim128(
            task.queries[query_index].query,
            hybrid_windows[query_index],
            backend.scoring_config,
        )
        var flat_query_scores = (
            exact_scores_for_hybrid_flat_only_index_dim128_with_flat_query(
                flat_queries[query_index],
                hybrid_windows[query_index],
                backend.scoring_config,
            )
        )

        require_score_lists_close(
            "candidate_window_hybrid_flat_nested_query",
            nested_scores,
            flat_scores,
            query_index,
        )
        require_score_lists_close(
            "candidate_window_hybrid_flat_flat_query",
            nested_scores,
            flat_query_scores,
            query_index,
        )


def benchmark_current_nested_score_mean_seconds(
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


def benchmark_candidate_window_build_mean_seconds(
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def build_once() capturing raises:
        bench_compiler.keep(
            build_candidate_window_hybrid_flat_index(
                snapshot,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(resolved_windows):
            query_index = 0

    var report = benchmark.run[build_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_flat_candidate_window_score_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read hybrid_windows: List[HybridFlatDim128Index],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            exact_scores_for_hybrid_flat_only_index_dim128(
                task.queries[query_index].query,
                hybrid_windows[query_index],
                backend.scoring_config,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_flat_candidate_window_score_flat_query_mean_seconds(
    read backend: ExactCpuBackend,
    read flat_queries: List[FlatQueryDim128],
    read hybrid_windows: List[HybridFlatDim128Index],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            exact_scores_for_hybrid_flat_only_index_dim128_with_flat_query(
                flat_queries[query_index],
                hybrid_windows[query_index],
                backend.scoring_config,
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_build_and_flat_score_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var hybrid_window = build_candidate_window_hybrid_flat_index(
            snapshot,
            resolved_windows[query_index],
        )
        bench_compiler.keep(
            exact_scores_for_hybrid_flat_only_index_dim128(
                task.queries[query_index].query,
                hybrid_window,
                backend.scoring_config,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_flat_query_build_and_flat_score_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        var hybrid_window = build_candidate_window_hybrid_flat_index(
            snapshot,
            resolved_windows[query_index],
        )
        bench_compiler.keep(
            exact_scores_for_hybrid_flat_only_index_dim128_with_flat_query(
                flat_query,
                hybrid_window,
                backend.scoring_config,
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
    mut measurements: List[CandidateFlatMeasurement],
    read task: JudgedTask,
    benchmark_kind: String,
    candidate_k: Int,
    mean_candidate_document_count: Float64,
    mean_candidate_vector_count: Float64,
    mean_seconds: Float64,
):
    measurements.append(
        CandidateFlatMeasurement(
            benchmark_kind.copy(),
            task.slice_name.copy(),
            candidate_k,
            len(task.queries),
            mean_candidate_document_count,
            mean_candidate_vector_count,
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
        "browsecomp_plus_gold_document_proxy_stage2_candidate_flat_compare",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_stage2_candidate_flat_compare"),
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
    var hybrid_windows = precompute_hybrid_candidate_windows(snapshot, resolved_windows)
    var flat_queries = precompute_flat_queries(task)
    var measurements = List[CandidateFlatMeasurement]()
    var total_candidate_document_count = Float64(0.0)
    var total_candidate_vector_count = Float64(0.0)

    for resolved in resolved_windows:
        total_candidate_document_count += Float64(len(resolved.documents))
        total_candidate_vector_count += Float64(resolved.vector_count)

    var mean_candidate_document_count = (
        total_candidate_document_count / Float64(len(resolved_windows))
    )
    var mean_candidate_vector_count = (
        total_candidate_vector_count / Float64(len(resolved_windows))
    )

    require_flat_candidate_window_scores_match(
        backend,
        task,
        snapshot,
        resolved_windows,
        hybrid_windows,
        flat_queries,
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
        " mean_candidate_docs= ",
        mean_candidate_document_count,
        " mean_candidate_vectors= ",
        mean_candidate_vector_count,
    )
    print("")

    print("validated candidate-window flat score equivalence against current stage-2 scorer")
    print("")

    print("== current nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "current_nested_score_resolved_candidate_window_for_cpu",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_current_nested_score_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )
    print("== build_candidate_window_hybrid_flat_index ==")
    append_measurement(
        measurements,
        task,
        "build_candidate_window_hybrid_flat_index",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_candidate_window_build_mean_seconds(
            snapshot,
            resolved_windows,
        ),
    )
    print("== prebuilt flat candidate-window score ==")
    append_measurement(
        measurements,
        task,
        "prebuilt_flat_candidate_window_score",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_flat_candidate_window_score_mean_seconds(
            backend,
            task,
            hybrid_windows,
        ),
    )
    print("== prebuilt flat candidate-window score with flat query ==")
    append_measurement(
        measurements,
        task,
        "prebuilt_flat_candidate_window_score_with_flat_query",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_flat_candidate_window_score_flat_query_mean_seconds(
            backend,
            flat_queries,
            hybrid_windows,
        ),
    )
    print("== build candidate-window hybrid flat index and score ==")
    append_measurement(
        measurements,
        task,
        "build_candidate_window_hybrid_flat_index_and_score",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_build_and_flat_score_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )
    print("== build flat query, build candidate-window hybrid flat index, and score ==")
    append_measurement(
        measurements,
        task,
        "build_flat_query_build_candidate_window_hybrid_flat_index_and_score",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_flat_query_build_and_flat_score_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root
        / "profile_document_proxy_stage2_candidate_flat_compare_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
