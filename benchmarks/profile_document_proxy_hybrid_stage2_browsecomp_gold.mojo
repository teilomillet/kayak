import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler
from std.pathlib import Path

from std.collections import List
from std.os import makedirs

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
    build_hybrid_flat_dim128_index,
    candidate_generation_for_plan,
    document_proxy_search_plan,
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_one_segment_collection_mirror,
    exact_score_for_hybrid_flat_document_dim128,
    exact_score_for_hybrid_flat_document_dim128_with_flat_query,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import FlatQueryDim128
from kayak.eval import JudgedTask
from kayak.index import HybridFlatDim128Index
from kayak.numeric import ScoreScalar
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    ResolvedCandidateWindow,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)


struct HybridStage2Measurement(Copyable):
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
    mut out: String, read measurement: HybridStage2Measurement
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
    path: Path, measurements: List[HybridStage2Measurement]
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


def score_candidate_set_with_hybrid_index(
    read query_index: Int,
    read task: JudgedTask,
    read candidate_set: CandidateSet,
    read hybrid_index: HybridFlatDim128Index,
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()

    for hit in candidate_set.hits:
        scores.append(
            exact_score_for_hybrid_flat_document_dim128(
                task.queries[query_index].query,
                hybrid_index,
                hit.document_index,
            )
        )

    return scores^


def score_candidate_set_with_hybrid_index_flat_query(
    read flat_query: FlatQueryDim128,
    read candidate_set: CandidateSet,
    read hybrid_index: HybridFlatDim128Index,
) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()

    for hit in candidate_set.hits:
        scores.append(
            exact_score_for_hybrid_flat_document_dim128_with_flat_query(
                flat_query,
                hybrid_index,
                hit.document_index,
            )
        )

    return scores^


def precompute_flat_queries(read task: JudgedTask) raises -> List[FlatQueryDim128]:
    var flat_queries = List[FlatQueryDim128]()

    for judged_query in task.queries:
        flat_queries.append(build_flat_query_dim128(judged_query.query))

    return flat_queries^


def benchmark_nested_score_mean_seconds(
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


def benchmark_hybrid_score_mean_seconds(
    read task: JudgedTask,
    read candidate_sets: List[CandidateSet],
    read hybrid_index: HybridFlatDim128Index,
) raises -> Float64:
    var query_index = 0

    def score_once() capturing:
        bench_compiler.keep(
            score_candidate_set_with_hybrid_index(
                query_index,
                task,
                candidate_sets[query_index],
                hybrid_index,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def benchmark_hybrid_score_flat_query_mean_seconds(
    read candidate_sets: List[CandidateSet],
    read flat_queries: List[FlatQueryDim128],
    read hybrid_index: HybridFlatDim128Index,
) raises -> Float64:
    var query_index = 0

    def score_once() capturing:
        bench_compiler.keep(
            score_candidate_set_with_hybrid_index_flat_query(
                flat_queries[query_index],
                candidate_sets[query_index],
                hybrid_index,
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def append_measurement(
    mut measurements: List[HybridStage2Measurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    benchmark_kind: String,
    candidate_k: Int,
    mean_seconds: Float64,
):
    measurements.append(
        HybridStage2Measurement(
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
        "browsecomp_plus_gold_document_proxy_hybrid_stage2",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_hybrid_stage2"),
        cache.stored_index,
        proxy_budget,
    )
    var plan = document_proxy_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    var hybrid_index = build_hybrid_flat_dim128_index(
        snapshot.segments[0].stored_index.index
    )
    var candidate_sets = precompute_candidate_sets(
        backend,
        task,
        snapshot,
        plan,
    )
    var resolved_windows = precompute_resolved_windows(snapshot, candidate_sets)
    var flat_queries = precompute_flat_queries(task)
    var measurements = List[HybridStage2Measurement]()

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

    print("== nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "nested_score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_nested_score_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )
    print("== hybrid score_candidate_set_with_hybrid_index ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "hybrid_score_candidate_set_with_hybrid_index",
        candidate_k,
        benchmark_hybrid_score_mean_seconds(
            task,
            candidate_sets,
            hybrid_index,
        ),
    )
    print("== hybrid score_candidate_set_with_hybrid_index_flat_query ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "hybrid_score_candidate_set_with_hybrid_index_flat_query",
        candidate_k,
        benchmark_hybrid_score_flat_query_mean_seconds(
            candidate_sets,
            flat_queries,
            hybrid_index,
        ),
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root / "profile_document_proxy_hybrid_stage2_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
