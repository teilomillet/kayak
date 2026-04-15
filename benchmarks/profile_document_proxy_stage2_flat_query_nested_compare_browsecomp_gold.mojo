import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.algorithm.backend.cpu.parallelize import sync_parallelize
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
from kayak.index import PackedIndex
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    ResolvedCandidateWindow,
    build_resolved_candidate_boundaries,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)
from kayak.scoring import dot_product_dim128_flat_at
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.maxsim import choose_parallel_work_item_count_for_shape


struct FlatQueryNestedMeasurement(Copyable):
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
    path: Path, measurements: List[FlatQueryNestedMeasurement]
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


def flat_query_exact_score_for_document_dim128(
    read query: FlatQueryDim128,
    read index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()

    for query_index in range(query.vector_count):
        var query_offset = query_index * COLBERT_VECTOR_DIM
        var best_similarity = min_score_scalar()

        for token_index in range(start, stop):
            var similarity = dot_product_dim128_flat_at(
                index.token_vectors[token_index],
                query.token_values,
                query_offset,
            )

            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def flat_query_score_resolved_candidate_window_for_cpu(
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
            scores_ptr[document_index] = flat_query_exact_score_for_document_dim128(
                query,
                snapshot.segments[resolved_document.segment_index].stored_index.index,
                resolved_document.document_index,
            )
        return scores^

    var boundaries = build_resolved_candidate_boundaries(
        resolved,
        work_item_count,
    )

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = flat_query_exact_score_for_document_dim128(
                query,
                snapshot.segments[resolved_document.segment_index].stored_index.index,
                resolved_document.document_index,
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


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


def require_flat_query_nested_scores_match(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
    read flat_queries: List[FlatQueryDim128],
) raises:
    for query_index in range(len(task.queries)):
        var current_scores = score_resolved_candidate_window_for_cpu(
            backend,
            task.queries[query_index].query,
            snapshot,
            resolved_windows[query_index],
        )
        var flat_query_scores = flat_query_score_resolved_candidate_window_for_cpu(
            backend,
            flat_queries[query_index],
            snapshot,
            resolved_windows[query_index],
        )
        require_score_lists_close(
            "flat_query_nested_stage2",
            current_scores,
            flat_query_scores,
            query_index,
        )


def benchmark_current_mean_seconds(
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


def benchmark_build_flat_query_mean_seconds(
    read task: JudgedTask
) raises -> Float64:
    var query_index = 0

    def build_once() capturing raises:
        bench_compiler.keep(build_flat_query_dim128(task.queries[query_index].query))
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[build_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_prebuilt_flat_query_nested_mean_seconds(
    read backend: ExactCpuBackend,
    read flat_queries: List[FlatQueryDim128],
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            flat_query_score_resolved_candidate_window_for_cpu(
                backend,
                flat_queries[query_index],
                snapshot,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_build_flat_query_and_nested_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        bench_compiler.keep(
            flat_query_score_resolved_candidate_window_for_cpu(
                backend,
                flat_query,
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


def append_measurement(
    mut measurements: List[FlatQueryNestedMeasurement],
    read task: JudgedTask,
    benchmark_kind: String,
    candidate_k: Int,
    mean_candidate_document_count: Float64,
    mean_candidate_vector_count: Float64,
    mean_seconds: Float64,
):
    measurements.append(
        FlatQueryNestedMeasurement(
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
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_document_proxy_stage2_flat_query_nested_compare",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_stage2_flat_query_nested_compare"),
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
    var flat_queries = precompute_flat_queries(task)
    var measurements = List[FlatQueryNestedMeasurement]()
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

    require_flat_query_nested_scores_match(
        backend,
        task,
        snapshot,
        resolved_windows,
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

    print("validated flat-query nested stage-2 score equivalence against current scorer")
    print("")

    print("== current score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "current_score_resolved_candidate_window_for_cpu",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_current_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )
    print("== build_flat_query_dim128 ==")
    append_measurement(
        measurements,
        task,
        "build_flat_query_dim128",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_build_flat_query_mean_seconds(task),
    )
    print("== prebuilt flat-query nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "prebuilt_flat_query_nested_score_resolved_candidate_window_for_cpu",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_prebuilt_flat_query_nested_mean_seconds(
            backend,
            flat_queries,
            snapshot,
            resolved_windows,
        ),
    )
    print("== build_flat_query_dim128 and nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "build_flat_query_dim128_and_nested_score_resolved_candidate_window_for_cpu",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_build_flat_query_and_nested_mean_seconds(
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
        / "profile_document_proxy_stage2_flat_query_nested_compare_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
