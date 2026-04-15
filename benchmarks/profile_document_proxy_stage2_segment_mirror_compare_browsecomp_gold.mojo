import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.sys.info import simd_width_of

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
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import FlatQueryDim128
from kayak.eval import JudgedTask
from kayak.index import HybridFlatDim128Index
from kayak.numeric import (
    VECTOR_SCALAR_NAME,
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_score_scalar,
)
from kayak.planning.candidate_set import CandidateSet
from kayak.planning.exact_stage import (
    ResolvedCandidateWindow,
    build_resolved_candidate_boundaries,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at
from kayak.scoring.maxsim import choose_parallel_work_item_count_for_shape

comptime STAGE2_BENCH_MIN_SECONDS = 0.05
comptime STAGE2_BENCH_MAX_SECONDS = 0.25
comptime STAGE2_BENCH_MAX_ITERS = 128
comptime STAGE2_BUILD_MIN_SECONDS = 0.01
comptime STAGE2_BUILD_MAX_SECONDS = 0.10
comptime STAGE2_BUILD_MAX_ITERS = 20


struct SegmentMirrorStage2Measurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var benchmark_kind: String
    var candidate_k: Int
    var segment_count: Int
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
        segment_count: Int,
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
        self.segment_count = segment_count
        self.document_count = document_count
        self.query_vector_count = query_vector_count
        self.nominal_document_vector_count = nominal_document_vector_count
        self.vector_dim = vector_dim
        self.mean_seconds = mean_seconds


def append_measurement_tsv_line(
    mut out: String, read measurement: SegmentMirrorStage2Measurement
):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.benchmark_kind
    out += "\t"
    out += String(measurement.candidate_k)
    out += "\t"
    out += String(measurement.segment_count)
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
    path: Path, measurements: List[SegmentMirrorStage2Measurement]
) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tbenchmark_kind\tcandidate_k\tsegment_count\tdocument_count\tquery_vector_count\tnominal_document_vector_count\tvector_dim\tmean_seconds\n"

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


def precompute_flat_queries(read task: JudgedTask) raises -> List[FlatQueryDim128]:
    var flat_queries = List[FlatQueryDim128]()

    for judged_query in task.queries:
        flat_queries.append(build_flat_query_dim128(judged_query.query))

    return flat_queries^


def build_segment_hybrid_flat_dim128_mirrors(
    read snapshot: ResolvedCollectionSnapshot
) raises -> List[HybridFlatDim128Index]:
    var mirrors = List[HybridFlatDim128Index]()

    for segment in snapshot.segments:
        mirrors.append(build_hybrid_flat_dim128_index(segment.stored_index.index))

    return mirrors^


def exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4(
    read query: FlatQueryDim128,
    read hybrid_index: HybridFlatDim128Index,
    document_index: Int,
) -> ScoreScalar:
    if VECTOR_SCALAR_NAME != "Float32":
        return exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4_fallback(
            query,
            hybrid_index,
            document_index,
        )

    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        return exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4_fallback(
            query,
            hybrid_index,
            document_index,
        )

    var start_vector = hybrid_index.doc_offsets[document_index]
    var stop_vector = hybrid_index.doc_offsets[document_index + 1]
    var start_offset = start_vector * COLBERT_VECTOR_DIM
    var document_vector_count = stop_vector - start_vector
    var total = zero_score_scalar()
    var query_index = 0
    var query_values_ptr = query.token_values.unsafe_ptr()
    var token_values_ptr = hybrid_index.token_values.unsafe_ptr() + start_offset

    while query_index + 3 < query.vector_count:
        var query_offset0 = query_index * COLBERT_VECTOR_DIM
        var query_offset1 = (query_index + 1) * COLBERT_VECTOR_DIM
        var query_offset2 = (query_index + 2) * COLBERT_VECTOR_DIM
        var query_offset3 = (query_index + 3) * COLBERT_VECTOR_DIM
        var query_ptr0 = query_values_ptr + query_offset0
        var query_ptr1 = query_values_ptr + query_offset1
        var query_ptr2 = query_values_ptr + query_offset2
        var query_ptr3 = query_values_ptr + query_offset3
        var best_similarity0 = min_score_scalar()
        var best_similarity1 = min_score_scalar()
        var best_similarity2 = min_score_scalar()
        var best_similarity3 = min_score_scalar()

        for token_index in range(document_vector_count):
            var token_ptr = token_values_ptr + (token_index * COLBERT_VECTOR_DIM)
            var accum0 = SIMD[DType.float32, width](0.0)
            var accum1 = SIMD[DType.float32, width](0.0)
            var accum2 = SIMD[DType.float32, width](0.0)
            var accum3 = SIMD[DType.float32, width](0.0)

            for dim in range(0, COLBERT_VECTOR_DIM, width):
                var token_chunk = (token_ptr + dim).load[width=width]()
                accum0 += (query_ptr0 + dim).load[width=width]() * token_chunk
                accum1 += (query_ptr1 + dim).load[width=width]() * token_chunk
                accum2 += (query_ptr2 + dim).load[width=width]() * token_chunk
                accum3 += (query_ptr3 + dim).load[width=width]() * token_chunk

            var similarity0 = ScoreScalar(accum0.reduce_add()[0])
            var similarity1 = ScoreScalar(accum1.reduce_add()[0])
            var similarity2 = ScoreScalar(accum2.reduce_add()[0])
            var similarity3 = ScoreScalar(accum3.reduce_add()[0])

            if similarity0 > best_similarity0:
                best_similarity0 = similarity0
            if similarity1 > best_similarity1:
                best_similarity1 = similarity1
            if similarity2 > best_similarity2:
                best_similarity2 = similarity2
            if similarity3 > best_similarity3:
                best_similarity3 = similarity3
        total += best_similarity0
        total += best_similarity1
        total += best_similarity2
        total += best_similarity3
        query_index += 4

    while query_index < query.vector_count:
        var query_offset = query_index * COLBERT_VECTOR_DIM
        var best_similarity = min_score_scalar()

        for token_index in range(document_vector_count):
            var similarity = dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                hybrid_index.token_values,
                start_offset + (token_index * COLBERT_VECTOR_DIM),
            )
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity
        query_index += 1

    return total


def exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4_fallback(
    read query: FlatQueryDim128,
    read hybrid_index: HybridFlatDim128Index,
    document_index: Int,
) -> ScoreScalar:
    var start_vector = hybrid_index.doc_offsets[document_index]
    var stop_vector = hybrid_index.doc_offsets[document_index + 1]
    var start_offset = start_vector * COLBERT_VECTOR_DIM
    var document_vector_count = stop_vector - start_vector
    var total = zero_score_scalar()

    for query_index in range(query.vector_count):
        var query_offset = query_index * COLBERT_VECTOR_DIM
        var best_similarity = min_score_scalar()

        for token_index in range(document_vector_count):
            var similarity = dot_product_dim128_flat_pair_at(
                query.token_values,
                query_offset,
                hybrid_index.token_values,
                start_offset + (token_index * COLBERT_VECTOR_DIM),
            )
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity

    return total


def segment_mirror_tiled4_score_resolved_candidate_window_for_cpu(
    read backend: ExactCpuBackend,
    read query: FlatQueryDim128,
    read mirrors: List[HybridFlatDim128Index],
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
                exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4(
                    query,
                    mirrors[resolved_document.segment_index],
                    resolved_document.document_index,
                )
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
            scores_ptr[document_index] = (
                exact_score_for_hybrid_flat_document_dim128_with_flat_query_tiled4(
                    query,
                    mirrors[resolved_document.segment_index],
                    resolved_document.document_index,
                )
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


def require_segment_mirror_scores_match(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
    read flat_queries: List[FlatQueryDim128],
    read mirrors: List[HybridFlatDim128Index],
) raises:
    for query_index in range(len(task.queries)):
        var current_scores = score_resolved_candidate_window_for_cpu(
            backend,
            task.queries[query_index].query,
            snapshot,
            resolved_windows[query_index],
        )
        var mirror_scores = segment_mirror_tiled4_score_resolved_candidate_window_for_cpu(
            backend,
            flat_queries[query_index],
            mirrors,
            resolved_windows[query_index],
        )
        require_score_lists_close(
            "segment_mirror_tiled4_stage2",
            current_scores,
            mirror_scores,
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

    var report = benchmark.run[score_once](
        num_warmup_iters=1,
        max_iters=STAGE2_BENCH_MAX_ITERS,
        min_runtime_secs=STAGE2_BENCH_MIN_SECONDS,
        max_runtime_secs=STAGE2_BENCH_MAX_SECONDS,
        max_batch_size=1,
    )
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

    var report = benchmark.run[build_once](
        num_warmup_iters=1,
        max_iters=STAGE2_BENCH_MAX_ITERS,
        min_runtime_secs=STAGE2_BENCH_MIN_SECONDS,
        max_runtime_secs=STAGE2_BENCH_MAX_SECONDS,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def benchmark_build_segment_mirrors_mean_seconds(
    read snapshot: ResolvedCollectionSnapshot
) raises -> Float64:
    def build_once() capturing raises:
        bench_compiler.keep(build_segment_hybrid_flat_dim128_mirrors(snapshot))

    var report = benchmark.run[build_once](
        num_warmup_iters=1,
        max_iters=STAGE2_BUILD_MAX_ITERS,
        min_runtime_secs=STAGE2_BUILD_MIN_SECONDS,
        max_runtime_secs=STAGE2_BUILD_MAX_SECONDS,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def benchmark_prebuilt_segment_mirror_tiled4_mean_seconds(
    read backend: ExactCpuBackend,
    read flat_queries: List[FlatQueryDim128],
    read mirrors: List[HybridFlatDim128Index],
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            segment_mirror_tiled4_score_resolved_candidate_window_for_cpu(
                backend,
                flat_queries[query_index],
                mirrors,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    var report = benchmark.run[score_once](
        num_warmup_iters=1,
        max_iters=STAGE2_BENCH_MAX_ITERS,
        min_runtime_secs=STAGE2_BENCH_MIN_SECONDS,
        max_runtime_secs=STAGE2_BENCH_MAX_SECONDS,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def benchmark_build_flat_query_and_prebuilt_segment_mirror_tiled4_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read mirrors: List[HybridFlatDim128Index],
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        bench_compiler.keep(
            segment_mirror_tiled4_score_resolved_candidate_window_for_cpu(
                backend,
                flat_query,
                mirrors,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once](
        num_warmup_iters=1,
        max_iters=STAGE2_BENCH_MAX_ITERS,
        min_runtime_secs=STAGE2_BENCH_MIN_SECONDS,
        max_runtime_secs=STAGE2_BENCH_MAX_SECONDS,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def benchmark_build_segment_mirrors_and_score_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read flat_queries: List[FlatQueryDim128],
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var mirrors = build_segment_hybrid_flat_dim128_mirrors(snapshot)
        bench_compiler.keep(
            segment_mirror_tiled4_score_resolved_candidate_window_for_cpu(
                backend,
                flat_queries[query_index],
                mirrors,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(flat_queries):
            query_index = 0

    var report = benchmark.run[score_once](
        num_warmup_iters=1,
        max_iters=STAGE2_BENCH_MAX_ITERS,
        min_runtime_secs=STAGE2_BENCH_MIN_SECONDS,
        max_runtime_secs=STAGE2_BENCH_MAX_SECONDS,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def benchmark_build_segment_mirrors_build_flat_query_and_score_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var mirrors = build_segment_hybrid_flat_dim128_mirrors(snapshot)
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        bench_compiler.keep(
            segment_mirror_tiled4_score_resolved_candidate_window_for_cpu(
                backend,
                flat_query,
                mirrors,
                resolved_windows[query_index],
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once](
        num_warmup_iters=1,
        max_iters=STAGE2_BENCH_MAX_ITERS,
        min_runtime_secs=STAGE2_BENCH_MIN_SECONDS,
        max_runtime_secs=STAGE2_BENCH_MAX_SECONDS,
        max_batch_size=1,
    )
    report.print()
    print("")
    return report.mean()


def append_measurement(
    mut measurements: List[SegmentMirrorStage2Measurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    benchmark_kind: String,
    candidate_k: Int,
    mean_seconds: Float64,
):
    measurements.append(
        SegmentMirrorStage2Measurement(
            dataset_name.copy(),
            task.slice_name.copy(),
            benchmark_kind.copy(),
            candidate_k,
            len(snapshot.segments),
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
        "browsecomp_plus_gold_document_proxy_stage2_segment_mirror_compare",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_stage2_segment_mirror_compare"),
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
    var flat_queries = precompute_flat_queries(task)
    var mirrors = build_segment_hybrid_flat_dim128_mirrors(snapshot)
    var measurements = List[SegmentMirrorStage2Measurement]()

    require_segment_mirror_scores_match(
        backend,
        task,
        snapshot,
        resolved_windows,
        flat_queries,
        mirrors,
    )

    print(
        "dataset= BrowseComp-Plus Gold  slice= ",
        task.slice_name,
        " segments= ",
        len(snapshot.segments),
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
    print(
        "validated segment-mirror tiled4 score equivalence against current scorer on resolved candidate windows"
    )
    print("")

    print("== current score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "current_score_resolved_candidate_window_for_cpu",
        candidate_k,
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
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "build_flat_query_dim128",
        candidate_k,
        benchmark_build_flat_query_mean_seconds(task),
    )

    print("== build_segment_hybrid_flat_dim128_mirrors ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "build_segment_hybrid_flat_dim128_mirrors",
        candidate_k,
        benchmark_build_segment_mirrors_mean_seconds(snapshot),
    )

    print("== prebuilt segment-mirror tiled4 score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "prebuilt_segment_mirror_tiled4_score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_prebuilt_segment_mirror_tiled4_mean_seconds(
            backend,
            flat_queries,
            mirrors,
            resolved_windows,
        ),
    )

    print("== build_flat_query_dim128 and prebuilt segment-mirror tiled4 score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "build_flat_query_dim128_and_prebuilt_segment_mirror_tiled4_score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_build_flat_query_and_prebuilt_segment_mirror_tiled4_mean_seconds(
            backend,
            task,
            mirrors,
            resolved_windows,
        ),
    )

    print("== build_segment_hybrid_flat_dim128_mirrors and score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "build_segment_hybrid_flat_dim128_mirrors_and_score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_build_segment_mirrors_and_score_mean_seconds(
            backend,
            task,
            snapshot,
            flat_queries,
            resolved_windows,
        ),
    )

    print("== build_segment_hybrid_flat_dim128_mirrors, build_flat_query_dim128, and score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        "BrowseComp-Plus Gold",
        task,
        snapshot,
        "build_segment_hybrid_flat_dim128_mirrors_build_flat_query_dim128_and_score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_build_segment_mirrors_build_flat_query_and_score_mean_seconds(
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
        / "profile_document_proxy_stage2_segment_mirror_compare_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
