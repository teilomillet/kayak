import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler
import kayak.planning.exact_stage_segment_mirror as segment_mirror_exact_stage

from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.sys.info import simd_width_of

from kayak import (
    CollectionHit,
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
    build_hybrid_flat_dim128_index,
    candidate_generation_for_plan,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
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
    exact_rerank_candidates_for_plan,
    resolve_candidate_window,
    score_resolved_candidate_window_for_cpu,
)
from kayak.planning.topk import insert_descending_collection_hit
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.dot128_flat import dot_product_dim128_flat_pair_at
from kayak.scoring.maxsim import choose_parallel_work_item_count_for_shape


comptime CENTROID_HEAD_POSTING_CAP = 16
comptime STAGE2_BENCH_MIN_SECONDS = 0.05
comptime STAGE2_BENCH_MAX_SECONDS = 0.25
comptime STAGE2_BENCH_MAX_ITERS = 128
comptime STAGE2_BUILD_MIN_SECONDS = 0.01
comptime STAGE2_BUILD_MAX_SECONDS = 0.10
comptime STAGE2_BUILD_MAX_ITERS = 20


struct SegmentMirrorExactStageMeasurement(Copyable):
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
    mut out: String, read measurement: SegmentMirrorExactStageMeasurement
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
    path: Path, measurements: List[SegmentMirrorExactStageMeasurement]
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


def precompute_flat_queries(read task: JudgedTask) raises -> List[FlatQueryDim128]:
    var flat_queries = List[FlatQueryDim128]()

    for judged_query in task.queries:
        flat_queries.append(build_flat_query_dim128(judged_query.query))

    return flat_queries^


def build_segment_hybrid_flat_dim128_mirrors(
    read snapshot: ResolvedCollectionSnapshot
) raises -> List[HybridFlatDim128Index]:
    return segment_mirror_exact_stage.build_segment_hybrid_flat_dim128_mirrors(
        snapshot
    )


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
    return segment_mirror_exact_stage.score_resolved_candidate_window_for_cpu_with_segment_mirrors_dim128_tiled4(
        backend,
        query,
        mirrors,
        resolved,
    )


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
            "segment_mirror_tiled4_exact_stage",
            current_scores,
            mirror_scores,
            query_index,
        )


def assemble_topk_hits(
    read candidate_set: CandidateSet,
    read resolved_window: ResolvedCandidateWindow,
    read scores: List[ScoreScalar],
    final_k: Int,
) -> List[CollectionHit]:
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


def exact_rerank_segment_mirror_reference(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read flat_query: FlatQueryDim128,
    read mirrors: List[HybridFlatDim128Index],
    read candidate_set: CandidateSet,
    read resolved_window: ResolvedCandidateWindow,
    final_k: Int,
) raises -> List[CollectionHit]:
    _ = flat_query
    _ = resolved_window
    return segment_mirror_exact_stage.exact_rerank_candidates_for_plan_with_segment_mirrors(
        backend,
        query,
        snapshot,
        mirrors,
        candidate_set.hits,
        final_k,
    ).final_hits.copy()


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


def benchmark_exact_rerank_current_mean_seconds(
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


def benchmark_exact_rerank_prebuilt_segment_mirror_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read flat_queries: List[FlatQueryDim128],
    read mirrors: List[HybridFlatDim128Index],
    read candidate_sets: List[CandidateSet],
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            exact_rerank_segment_mirror_reference(
                backend,
                task.queries[query_index].query,
                snapshot,
                flat_queries[query_index],
                mirrors,
                candidate_sets[query_index],
                resolved_windows[query_index],
                task.k,
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
    mut measurements: List[SegmentMirrorExactStageMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    generator_kind: String,
    benchmark_kind: String,
    candidate_k: Int,
    mean_seconds: Float64,
):
    measurements.append(
        SegmentMirrorExactStageMeasurement(
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
    mut measurements: List[SegmentMirrorExactStageMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    candidate_k: Int,
    read backend: ExactCpuBackend,
    read mirrors: List[HybridFlatDim128Index],
) raises:
    var candidate_sets = precompute_candidate_sets(
        backend,
        task,
        snapshot,
        plan,
    )
    var resolved_windows = precompute_resolved_windows(snapshot, candidate_sets)
    var flat_queries = precompute_flat_queries(task)

    require_segment_mirror_scores_match(
        backend,
        task,
        snapshot,
        resolved_windows,
        flat_queries,
        mirrors,
    )

    print("== ", plan.candidate_generator.kind, " current_score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "current_score_resolved_candidate_window_for_cpu",
        candidate_k,
        benchmark_current_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )
    print("== ", plan.candidate_generator.kind, " build_flat_query_dim128 ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "build_flat_query_dim128",
        candidate_k,
        benchmark_build_flat_query_mean_seconds(task),
    )
    print("== ", plan.candidate_generator.kind, " build_segment_hybrid_flat_dim128_mirrors ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "build_segment_hybrid_flat_dim128_mirrors",
        candidate_k,
        benchmark_build_segment_mirrors_mean_seconds(snapshot),
    )
    print("== ", plan.candidate_generator.kind, " prebuilt_segment_mirror_tiled4_score ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "prebuilt_segment_mirror_tiled4_score",
        candidate_k,
        benchmark_prebuilt_segment_mirror_tiled4_mean_seconds(
            backend,
            flat_queries,
            mirrors,
            resolved_windows,
        ),
    )
    print("== ", plan.candidate_generator.kind, " build_flat_query_plus_prebuilt_segment_mirror_tiled4_score ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "build_flat_query_plus_prebuilt_segment_mirror_tiled4_score",
        candidate_k,
        benchmark_build_flat_query_and_prebuilt_segment_mirror_tiled4_mean_seconds(
            backend,
            task,
            mirrors,
            resolved_windows,
        ),
    )
    print("== ", plan.candidate_generator.kind, " build_segment_mirrors_and_score ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "build_segment_mirrors_and_score",
        candidate_k,
        benchmark_build_segment_mirrors_and_score_mean_seconds(
            backend,
            task,
            snapshot,
            flat_queries,
            resolved_windows,
        ),
    )
    print("== ", plan.candidate_generator.kind, " exact_rerank_current ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "exact_rerank_current",
        candidate_k,
        benchmark_exact_rerank_current_mean_seconds(
            backend,
            task,
            snapshot,
            candidate_sets,
        ),
    )
    print("== ", plan.candidate_generator.kind, " exact_rerank_prebuilt_segment_mirror_reference ==")
    append_measurement(
        measurements,
        dataset_name,
        task,
        snapshot,
        plan.candidate_generator.kind,
        "exact_rerank_prebuilt_segment_mirror_reference",
        candidate_k,
        benchmark_exact_rerank_prebuilt_segment_mirror_mean_seconds(
            backend,
            task,
            snapshot,
            flat_queries,
            mirrors,
            candidate_sets,
            resolved_windows,
        ),
    )
    print("")


def main() raises:
    var backend = ExactCpuBackend()
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var candidate_k = stage1_candidate_k(
        task,
        cache.stored_index.index.document_count,
    )
    var snapshot = load_profile_snapshot(
        "browsecomp_plus_gold_centroid_exact_stage_segment_mirror_compare",
        Path(".cache/kayak/browsecomp_plus_gold_centroid_exact_stage_segment_mirror_compare"),
        cache.stored_index,
    )
    var mirrors = build_segment_hybrid_flat_dim128_mirrors(snapshot)
    var measurements = List[SegmentMirrorExactStageMeasurement]()

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
        mirrors,
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
        mirrors,
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = (
        output_root
        / "profile_centroid_exact_stage_segment_mirror_compare_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
