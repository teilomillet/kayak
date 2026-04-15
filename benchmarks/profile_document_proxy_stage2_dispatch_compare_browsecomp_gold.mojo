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
    PackedIndex,
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
from kayak.contracts import EncodedQuery
from kayak.eval import JudgedTask
from kayak.numeric import ScoreScalar, min_score_scalar, zero_score_scalar
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    ResolvedCandidateWindow,
    build_resolved_candidate_boundaries,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)
from kayak.scoring.dot128 import dot_product_dim128
from kayak.scoring.maxsim import (
    choose_parallel_work_item_count_for_shape,
    exact_score_for_document_with_config,
)


struct DispatchCompareMeasurement(Copyable):
    var benchmark_kind: String
    var slice_name: String
    var candidate_k: Int
    var mean_seconds: Float64

    def __init__(
        out self,
        var benchmark_kind: String,
        var slice_name: String,
        candidate_k: Int,
        mean_seconds: Float64,
    ):
        self.benchmark_kind = benchmark_kind^
        self.slice_name = slice_name^
        self.candidate_k = candidate_k
        self.mean_seconds = mean_seconds


def write_measurements_tsv(
    path: Path, measurements: List[DispatchCompareMeasurement]
) raises:
    var lines = String()
    lines += "benchmark_kind\tslice_name\tcandidate_k\tmean_seconds\tthroughput_per_second\n"

    for measurement in measurements:
        lines += measurement.benchmark_kind
        lines += "\t"
        lines += measurement.slice_name
        lines += "\t"
        lines += String(measurement.candidate_k)
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


def legacy_score_resolved_candidate_window_for_cpu(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    query_index: Int,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved: ResolvedCandidateWindow,
) raises -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for _ in range(len(resolved.documents)):
        scores.append(zero_score_scalar())

    var query = task.queries[query_index].query.copy()
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
            scores_ptr[document_index] = exact_score_for_document_with_config(
                query,
                snapshot.segments[resolved_document.segment_index].stored_index.index,
                resolved_document.document_index,
                backend.scoring_config,
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
            scores_ptr[document_index] = exact_score_for_document_with_config(
                query,
                snapshot.segments[resolved_document.segment_index].stored_index.index,
                resolved_document.document_index,
                backend.scoring_config,
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def doc_outer_exact_score_for_document_dim128(
    read query: EncodedQuery,
    read index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var best_scores = List[ScoreScalar]()

    for _ in range(query.vector_count):
        best_scores.append(min_score_scalar())

    var best_scores_ptr = best_scores.unsafe_ptr()

    for token_index in range(start, stop):
        for query_token_index in range(query.vector_count):
            var similarity = dot_product_dim128(
                query.token_vectors[query_token_index],
                index.token_vectors[token_index],
            )
            if similarity > best_scores_ptr[query_token_index]:
                best_scores_ptr[query_token_index] = similarity

    var total = zero_score_scalar()
    for query_token_index in range(query.vector_count):
        total += best_scores_ptr[query_token_index]
    return total


def doc_outer_score_resolved_candidate_window_for_cpu(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    query_index: Int,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved: ResolvedCandidateWindow,
) raises -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for _ in range(len(resolved.documents)):
        scores.append(zero_score_scalar())

    var work_item_count = choose_parallel_work_item_count_for_shape(
        task.queries[query_index].query.vector_count,
        len(resolved.documents),
        resolved.vector_count,
        backend.scoring_config,
    )
    var scores_ptr = scores.unsafe_ptr()

    if work_item_count <= 1:
        for document_index in range(len(resolved.documents)):
            var resolved_document = resolved.documents[document_index].copy()
            scores_ptr[document_index] = doc_outer_exact_score_for_document_dim128(
                task.queries[query_index].query,
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
            scores_ptr[document_index] = doc_outer_exact_score_for_document_dim128(
                task.queries[query_index].query,
                snapshot.segments[resolved_document.segment_index].stored_index.index,
                resolved_document.document_index,
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


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


def benchmark_legacy_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            legacy_score_resolved_candidate_window_for_cpu(
                backend,
                task,
                query_index,
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


def benchmark_doc_outer_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            doc_outer_score_resolved_candidate_window_for_cpu(
                backend,
                task,
                query_index,
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


def main() raises:
    var backend = ExactCpuBackend()
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var candidate_k = task.k * 4
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_document_proxy_stage2_dispatch_compare",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_stage2_dispatch_compare"),
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
    var measurements = List[DispatchCompareMeasurement]()

    print(
        "dispatch compare dataset=BrowseComp-Plus Gold slice=",
        task.slice_name,
        " candidate_k=",
        candidate_k,
    )
    print("== current score_resolved_candidate_window_for_cpu ==")
    measurements.append(
        DispatchCompareMeasurement(
            "current_score_resolved_candidate_window_for_cpu",
            task.slice_name.copy(),
            candidate_k,
            benchmark_current_mean_seconds(
                backend,
                task,
                snapshot,
                resolved_windows,
            ),
        )
    )
    print("== legacy score_resolved_candidate_window_for_cpu ==")
    measurements.append(
        DispatchCompareMeasurement(
            "legacy_score_resolved_candidate_window_for_cpu",
            task.slice_name.copy(),
            candidate_k,
            benchmark_legacy_mean_seconds(
                backend,
                task,
                snapshot,
                resolved_windows,
            ),
        )
    )
    print("== doc_outer score_resolved_candidate_window_for_cpu ==")
    measurements.append(
        DispatchCompareMeasurement(
            "doc_outer_score_resolved_candidate_window_for_cpu",
            task.slice_name.copy(),
            candidate_k,
            benchmark_doc_outer_mean_seconds(
                backend,
                task,
                snapshot,
                resolved_windows,
            ),
        )
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root / "profile_document_proxy_stage2_dispatch_compare_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
