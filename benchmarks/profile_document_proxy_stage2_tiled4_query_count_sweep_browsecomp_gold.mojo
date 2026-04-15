import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    EncodedQuery,
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
from kayak.numeric import ScoreScalar, zero_score_scalar
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    ResolvedCandidateWindow,
    build_resolved_candidate_boundaries,
    resolve_candidate_window,
)
from kayak.scoring.maxsim import (
    choose_parallel_work_item_count_for_shape,
    exact_score_for_document_dim128,
    exact_score_for_document_dim128_flat_query_tiled4,
)

# Keep the expanded count sweep bounded so the benchmark stays practical while
# still cycling over the real query set multiple times per shape.
comptime SWEEP_PASSES = 16
comptime TARGET_QUERY_COUNT = 64


struct QueryCountSweepMeasurement(Copyable):
    var benchmark_kind: String
    var slice_name: String
    var query_vector_count: Int
    var candidate_k: Int
    var mean_candidate_document_count: Float64
    var mean_candidate_vector_count: Float64
    var mean_seconds: Float64

    def __init__(
        out self,
        var benchmark_kind: String,
        var slice_name: String,
        query_vector_count: Int,
        candidate_k: Int,
        mean_candidate_document_count: Float64,
        mean_candidate_vector_count: Float64,
        mean_seconds: Float64,
    ):
        self.benchmark_kind = benchmark_kind^
        self.slice_name = slice_name^
        self.query_vector_count = query_vector_count
        self.candidate_k = candidate_k
        self.mean_candidate_document_count = mean_candidate_document_count
        self.mean_candidate_vector_count = mean_candidate_vector_count
        self.mean_seconds = mean_seconds


def write_measurements_tsv(
    path: Path, measurements: List[QueryCountSweepMeasurement]
) raises:
    var lines = String()
    lines += "benchmark_kind\tslice_name\tquery_vector_count\tcandidate_k\tmean_candidate_document_count\tmean_candidate_vector_count\tmean_seconds\tthroughput_per_second\n"

    for measurement in measurements:
        lines += measurement.benchmark_kind
        lines += "\t"
        lines += measurement.slice_name
        lines += "\t"
        lines += String(measurement.query_vector_count)
        lines += "\t"
        lines += String(measurement.candidate_k)
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


def append_query_prefix_or_cycle(
    mut token_vectors: List[List[Float32]],
    read source: EncodedQuery,
    target_vector_count: Int,
):
    for query_index in range(target_vector_count):
        token_vectors.append(
            source.token_vectors[query_index % source.vector_count].copy()
        )


def build_query_variant(
    read source: EncodedQuery, target_vector_count: Int
) raises -> EncodedQuery:
    if target_vector_count <= 0:
        raise Error("query_vector_count must be positive")

    var token_vectors = List[List[Float32]]()
    append_query_prefix_or_cycle(token_vectors, source, target_vector_count)
    return EncodedQuery(token_vectors^)


def precompute_query_variants(
    read task: JudgedTask, query_vector_count: Int
) raises -> List[EncodedQuery]:
    var queries = List[EncodedQuery]()

    for judged_query in task.queries:
        queries.append(
            build_query_variant(judged_query.query, query_vector_count)
        )

    return queries^


def precompute_flat_query_variants(
    read task: JudgedTask, query_vector_count: Int
) raises -> List[FlatQueryDim128]:
    var flat_queries = List[FlatQueryDim128]()

    for judged_query in task.queries:
        flat_queries.append(
            build_flat_query_dim128(
                build_query_variant(judged_query.query, query_vector_count)
            )
        )

    return flat_queries^


def baseline_score_resolved_candidate_window_for_cpu_dim128(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved: ResolvedCandidateWindow,
) raises -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for _ in range(len(resolved.documents)):
        scores.append(zero_score_scalar())

    var work_item_count = choose_parallel_work_item_count_for_shape(
        query.vector_count,
        len(resolved.documents),
        resolved.vector_count,
        backend.scoring_config,
    )
    var scores_ptr = scores.unsafe_ptr()

    if work_item_count <= 1:
        for document_index in range(len(resolved.documents)):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = exact_score_for_document_dim128(
                query,
                snapshot.segments[resolved_document.segment_index].stored_index.index,
                resolved_document.document_index,
            )
        return scores^

    var boundaries = build_resolved_candidate_boundaries(
        resolved, work_item_count
    )

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = exact_score_for_document_dim128(
                query,
                snapshot.segments[resolved_document.segment_index].stored_index.index,
                resolved_document.document_index,
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def tiled4_score_resolved_candidate_window_for_cpu_dim128(
    read backend: ExactCpuBackend,
    read query: FlatQueryDim128,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved: ResolvedCandidateWindow,
) raises -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for _ in range(len(resolved.documents)):
        scores.append(zero_score_scalar())

    var work_item_count = choose_parallel_work_item_count_for_shape(
        query.vector_count,
        len(resolved.documents),
        resolved.vector_count,
        backend.scoring_config,
    )
    var scores_ptr = scores.unsafe_ptr()

    if work_item_count <= 1:
        for document_index in range(len(resolved.documents)):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = (
                exact_score_for_document_dim128_flat_query_tiled4(
                    query,
                    snapshot.segments[resolved_document.segment_index].stored_index.index,
                    resolved_document.document_index,
                )
            )
        return scores^

    var boundaries = build_resolved_candidate_boundaries(
        resolved, work_item_count
    )

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = (
                exact_score_for_document_dim128_flat_query_tiled4(
                    query,
                    snapshot.segments[resolved_document.segment_index].stored_index.index,
                    resolved_document.document_index,
                )
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def require_score_lists_close(
    benchmark_kind: String,
    read expected: List[ScoreScalar],
    read observed: List[ScoreScalar],
    query_vector_count: Int,
    query_index: Int,
) raises:
    if len(expected) != len(observed):
        raise Error(
            benchmark_kind
            + " score count mismatch for q="
            + String(query_vector_count)
            + " query="
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
                + " score mismatch for q="
                + String(query_vector_count)
                + " query="
                + String(query_index)
                + " document="
                + String(document_index)
                + " difference="
                + String(difference)
            )


def require_query_count_variant_scores_match(
    read backend: ExactCpuBackend,
    read queries: List[EncodedQuery],
    read flat_queries: List[FlatQueryDim128],
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
    query_vector_count: Int,
) raises:
    for query_index in range(len(queries)):
        var baseline_scores = baseline_score_resolved_candidate_window_for_cpu_dim128(
            backend,
            queries[query_index],
            snapshot,
            resolved_windows[query_index],
        )
        var tiled4_scores = tiled4_score_resolved_candidate_window_for_cpu_dim128(
            backend,
            flat_queries[query_index],
            snapshot,
            resolved_windows[query_index],
        )
        require_score_lists_close(
            "document_proxy_stage2_tiled4_query_count_sweep",
            baseline_scores,
            tiled4_scores,
            query_vector_count,
            query_index,
        )


def benchmark_baseline_mean_seconds(
    read backend: ExactCpuBackend,
    read queries: List[EncodedQuery],
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            baseline_score_resolved_candidate_window_for_cpu_dim128(
                backend,
                queries[query_index],
                snapshot,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(queries):
            query_index = 0

    # Small q counts need more iterations than large q counts; otherwise their
    # runtimes are too short and host jitter dominates the comparison.
    var iteration_scale = TARGET_QUERY_COUNT // queries[0].vector_count
    if iteration_scale < 1:
        iteration_scale = 1

    var report = benchmark.run[score_once](
        num_warmup_iters=0,
        max_iters=len(queries) * SWEEP_PASSES * iteration_scale,
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def benchmark_prebuilt_tiled4_mean_seconds(
    read backend: ExactCpuBackend,
    read flat_queries: List[FlatQueryDim128],
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            tiled4_score_resolved_candidate_window_for_cpu_dim128(
                backend,
                flat_queries[query_index],
                snapshot,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    var iteration_scale = TARGET_QUERY_COUNT // flat_queries[0].vector_count
    if iteration_scale < 1:
        iteration_scale = 1

    var report = benchmark.run[score_once](
        num_warmup_iters=0,
        max_iters=len(flat_queries) * SWEEP_PASSES * iteration_scale,
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def benchmark_build_flat_query_and_tiled4_mean_seconds(
    read backend: ExactCpuBackend,
    read queries: List[EncodedQuery],
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(queries[query_index])
        bench_compiler.keep(
            tiled4_score_resolved_candidate_window_for_cpu_dim128(
                backend,
                flat_query,
                snapshot,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(queries):
            query_index = 0

    var iteration_scale = TARGET_QUERY_COUNT // queries[0].vector_count
    if iteration_scale < 1:
        iteration_scale = 1

    var report = benchmark.run[score_once](
        num_warmup_iters=0,
        max_iters=len(queries) * SWEEP_PASSES * iteration_scale,
        min_runtime_secs=0.0,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def append_measurement(
    mut measurements: List[QueryCountSweepMeasurement],
    read task: JudgedTask,
    benchmark_kind: String,
    query_vector_count: Int,
    candidate_k: Int,
    mean_candidate_document_count: Float64,
    mean_candidate_vector_count: Float64,
    mean_seconds: Float64,
):
    measurements.append(
        QueryCountSweepMeasurement(
            benchmark_kind.copy(),
            task.slice_name.copy(),
            query_vector_count,
            candidate_k,
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
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_document_proxy_stage2_tiled4_query_count_sweep",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_stage2_tiled4_query_count_sweep"),
        cache.stored_index,
        task.nominal_document_vector_count,
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
    var measurements = List[QueryCountSweepMeasurement]()
    var total_candidate_document_count = Float64(0.0)
    var total_candidate_vector_count = Float64(0.0)
    var query_vector_counts = List[Int]()
    for query_vector_count in range(4, 65):
        query_vector_counts.append(query_vector_count)

    for resolved in resolved_windows:
        total_candidate_document_count += Float64(len(resolved.documents))
        total_candidate_vector_count += Float64(resolved.vector_count)

    var mean_candidate_document_count = (
        total_candidate_document_count / Float64(len(resolved_windows))
    )
    var mean_candidate_vector_count = (
        total_candidate_vector_count / Float64(len(resolved_windows))
    )

    print(
        "dataset= BrowseComp-Plus Gold  slice= ",
        task.slice_name,
        " docs= ",
        snapshot.snapshot.stats.document_count,
        " base_query_vectors= ",
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
    print("query_count sweep uses q<=32 prefixes and q>32 cyclic extensions on the same resolved candidate windows")
    print("")

    for query_vector_count in query_vector_counts:
        var queries = precompute_query_variants(task, query_vector_count)
        var flat_queries = precompute_flat_query_variants(task, query_vector_count)

        require_query_count_variant_scores_match(
            backend,
            queries,
            flat_queries,
            snapshot,
            resolved_windows,
            query_vector_count,
        )

        print("== q=", query_vector_count, " baseline nested dim128 ==")
        append_measurement(
            measurements,
            task,
            "baseline_nested_dim128",
            query_vector_count,
            candidate_k,
            mean_candidate_document_count,
            mean_candidate_vector_count,
            benchmark_baseline_mean_seconds(
                backend,
                queries,
                snapshot,
                resolved_windows,
            ),
        )

        print("== q=", query_vector_count, " prebuilt tiled4 flat-query ==")
        append_measurement(
            measurements,
            task,
            "prebuilt_tiled4_flat_query",
            query_vector_count,
            candidate_k,
            mean_candidate_document_count,
            mean_candidate_vector_count,
            benchmark_prebuilt_tiled4_mean_seconds(
                backend,
                flat_queries,
                snapshot,
                resolved_windows,
            ),
        )

        print("== q=", query_vector_count, " build_flat_query + tiled4 ==")
        append_measurement(
            measurements,
            task,
            "build_flat_query_and_tiled4",
            query_vector_count,
            candidate_k,
            mean_candidate_document_count,
            mean_candidate_vector_count,
            benchmark_build_flat_query_and_tiled4_mean_seconds(
                backend,
                queries,
                snapshot,
                resolved_windows,
            ),
        )
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root
        / "profile_document_proxy_stage2_tiled4_query_count_sweep_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
