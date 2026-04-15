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
from kayak.numeric import (
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
from kayak.scoring import dot_product_dim128_flat_at
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.maxsim import choose_parallel_work_item_count_for_shape


struct FlatQueryTiledKernelMeasurement(Copyable):
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
    path: Path, measurements: List[FlatQueryTiledKernelMeasurement]
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


def tiled4_flat_query_exact_score_for_document_dim128(
    read query: FlatQueryDim128,
    read index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        return flat_query_exact_score_for_document_dim128(
            query,
            index,
            document_index,
        )

    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()
    var query_index = 0
    var query_values_ptr = query.token_values.unsafe_ptr()

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

        for token_index in range(start, stop):
            var token_ptr = index.token_vectors[token_index].unsafe_ptr()
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

        for token_index in range(start, stop):
            var similarity = dot_product_dim128_flat_at(
                index.token_vectors[token_index],
                query.token_values,
                query_offset,
            )
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity
        query_index += 1

    return total


def tiled8_flat_query_exact_score_for_document_dim128(
    read query: FlatQueryDim128,
    read index: PackedIndex,
    document_index: Int,
) -> ScoreScalar:
    comptime width = simd_width_of[VectorScalar]()
    if COLBERT_VECTOR_DIM % width != 0:
        return flat_query_exact_score_for_document_dim128(
            query,
            index,
            document_index,
        )

    var start = index.doc_offsets[document_index]
    var stop = index.doc_offsets[document_index + 1]
    var total = zero_score_scalar()
    var query_index = 0
    var query_values_ptr = query.token_values.unsafe_ptr()

    while query_index + 7 < query.vector_count:
        var query_offset0 = query_index * COLBERT_VECTOR_DIM
        var query_offset1 = (query_index + 1) * COLBERT_VECTOR_DIM
        var query_offset2 = (query_index + 2) * COLBERT_VECTOR_DIM
        var query_offset3 = (query_index + 3) * COLBERT_VECTOR_DIM
        var query_offset4 = (query_index + 4) * COLBERT_VECTOR_DIM
        var query_offset5 = (query_index + 5) * COLBERT_VECTOR_DIM
        var query_offset6 = (query_index + 6) * COLBERT_VECTOR_DIM
        var query_offset7 = (query_index + 7) * COLBERT_VECTOR_DIM
        var query_ptr0 = query_values_ptr + query_offset0
        var query_ptr1 = query_values_ptr + query_offset1
        var query_ptr2 = query_values_ptr + query_offset2
        var query_ptr3 = query_values_ptr + query_offset3
        var query_ptr4 = query_values_ptr + query_offset4
        var query_ptr5 = query_values_ptr + query_offset5
        var query_ptr6 = query_values_ptr + query_offset6
        var query_ptr7 = query_values_ptr + query_offset7
        var best_similarity0 = min_score_scalar()
        var best_similarity1 = min_score_scalar()
        var best_similarity2 = min_score_scalar()
        var best_similarity3 = min_score_scalar()
        var best_similarity4 = min_score_scalar()
        var best_similarity5 = min_score_scalar()
        var best_similarity6 = min_score_scalar()
        var best_similarity7 = min_score_scalar()

        for token_index in range(start, stop):
            var token_ptr = index.token_vectors[token_index].unsafe_ptr()
            var accum0 = SIMD[DType.float32, width](0.0)
            var accum1 = SIMD[DType.float32, width](0.0)
            var accum2 = SIMD[DType.float32, width](0.0)
            var accum3 = SIMD[DType.float32, width](0.0)
            var accum4 = SIMD[DType.float32, width](0.0)
            var accum5 = SIMD[DType.float32, width](0.0)
            var accum6 = SIMD[DType.float32, width](0.0)
            var accum7 = SIMD[DType.float32, width](0.0)

            for dim in range(0, COLBERT_VECTOR_DIM, width):
                var token_chunk = (token_ptr + dim).load[width=width]()
                accum0 += (query_ptr0 + dim).load[width=width]() * token_chunk
                accum1 += (query_ptr1 + dim).load[width=width]() * token_chunk
                accum2 += (query_ptr2 + dim).load[width=width]() * token_chunk
                accum3 += (query_ptr3 + dim).load[width=width]() * token_chunk
                accum4 += (query_ptr4 + dim).load[width=width]() * token_chunk
                accum5 += (query_ptr5 + dim).load[width=width]() * token_chunk
                accum6 += (query_ptr6 + dim).load[width=width]() * token_chunk
                accum7 += (query_ptr7 + dim).load[width=width]() * token_chunk

            var similarity0 = ScoreScalar(accum0.reduce_add()[0])
            var similarity1 = ScoreScalar(accum1.reduce_add()[0])
            var similarity2 = ScoreScalar(accum2.reduce_add()[0])
            var similarity3 = ScoreScalar(accum3.reduce_add()[0])
            var similarity4 = ScoreScalar(accum4.reduce_add()[0])
            var similarity5 = ScoreScalar(accum5.reduce_add()[0])
            var similarity6 = ScoreScalar(accum6.reduce_add()[0])
            var similarity7 = ScoreScalar(accum7.reduce_add()[0])

            if similarity0 > best_similarity0:
                best_similarity0 = similarity0
            if similarity1 > best_similarity1:
                best_similarity1 = similarity1
            if similarity2 > best_similarity2:
                best_similarity2 = similarity2
            if similarity3 > best_similarity3:
                best_similarity3 = similarity3
            if similarity4 > best_similarity4:
                best_similarity4 = similarity4
            if similarity5 > best_similarity5:
                best_similarity5 = similarity5
            if similarity6 > best_similarity6:
                best_similarity6 = similarity6
            if similarity7 > best_similarity7:
                best_similarity7 = similarity7
        total += best_similarity0
        total += best_similarity1
        total += best_similarity2
        total += best_similarity3
        total += best_similarity4
        total += best_similarity5
        total += best_similarity6
        total += best_similarity7
        query_index += 8

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

        for token_index in range(start, stop):
            var token_ptr = index.token_vectors[token_index].unsafe_ptr()
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

        for token_index in range(start, stop):
            var similarity = dot_product_dim128_flat_at(
                index.token_vectors[token_index],
                query.token_values,
                query_offset,
            )
            if similarity > best_similarity:
                best_similarity = similarity

        total += best_similarity
        query_index += 1

    return total


def tiled4_flat_query_score_resolved_candidate_window_for_cpu(
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
                tiled4_flat_query_exact_score_for_document_dim128(
                    query,
                    snapshot.segments[resolved_document.segment_index].stored_index.index,
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
                tiled4_flat_query_exact_score_for_document_dim128(
                    query,
                    snapshot.segments[resolved_document.segment_index].stored_index.index,
                    resolved_document.document_index,
                )
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def tiled8_flat_query_score_resolved_candidate_window_for_cpu(
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
                tiled8_flat_query_exact_score_for_document_dim128(
                    query,
                    snapshot.segments[resolved_document.segment_index].stored_index.index,
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
                tiled8_flat_query_exact_score_for_document_dim128(
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


def require_tiled4_scores_match(
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
        var tiled4_scores = tiled4_flat_query_score_resolved_candidate_window_for_cpu(
            backend,
            flat_queries[query_index],
            snapshot,
            resolved_windows[query_index],
        )
        require_score_lists_close(
            "flat_query_tiled4_stage2",
            current_scores,
            tiled4_scores,
            query_index,
        )


def require_tiled8_scores_match(
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
        var tiled8_scores = tiled8_flat_query_score_resolved_candidate_window_for_cpu(
            backend,
            flat_queries[query_index],
            snapshot,
            resolved_windows[query_index],
        )
        require_score_lists_close(
            "flat_query_tiled8_stage2",
            current_scores,
            tiled8_scores,
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


def benchmark_prebuilt_tiled4_mean_seconds(
    read backend: ExactCpuBackend,
    read flat_queries: List[FlatQueryDim128],
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            tiled4_flat_query_score_resolved_candidate_window_for_cpu(
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


def benchmark_build_flat_query_and_tiled4_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        bench_compiler.keep(
            tiled4_flat_query_score_resolved_candidate_window_for_cpu(
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


def benchmark_prebuilt_tiled8_mean_seconds(
    read backend: ExactCpuBackend,
    read flat_queries: List[FlatQueryDim128],
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            tiled8_flat_query_score_resolved_candidate_window_for_cpu(
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


def benchmark_build_flat_query_and_tiled8_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_windows: List[ResolvedCandidateWindow],
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        bench_compiler.keep(
            tiled8_flat_query_score_resolved_candidate_window_for_cpu(
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
    mut measurements: List[FlatQueryTiledKernelMeasurement],
    read task: JudgedTask,
    benchmark_kind: String,
    candidate_k: Int,
    mean_candidate_document_count: Float64,
    mean_candidate_vector_count: Float64,
    mean_seconds: Float64,
):
    measurements.append(
        FlatQueryTiledKernelMeasurement(
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
        "browsecomp_plus_gold_document_proxy_stage2_flat_query_tiled4_compare",
        Path(".cache/kayak/browsecomp_plus_gold_document_proxy_stage2_flat_query_tiled4_compare"),
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
    var measurements = List[FlatQueryTiledKernelMeasurement]()
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

    require_tiled4_scores_match(
        backend,
        task,
        snapshot,
        resolved_windows,
        flat_queries,
    )
    require_tiled8_scores_match(
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

    print("validated flat-query tiled4 and tiled8 stage-2 score equivalence against current scorer")
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
    print("== prebuilt flat-query tiled4 nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "prebuilt_flat_query_tiled4_nested_score_resolved_candidate_window_for_cpu",
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
    print("== build_flat_query_dim128 and flat-query tiled4 nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "build_flat_query_dim128_and_flat_query_tiled4_nested_score_resolved_candidate_window_for_cpu",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_build_flat_query_and_tiled4_mean_seconds(
            backend,
            task,
            snapshot,
            resolved_windows,
        ),
    )
    print("== prebuilt flat-query tiled8 nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "prebuilt_flat_query_tiled8_nested_score_resolved_candidate_window_for_cpu",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_prebuilt_tiled8_mean_seconds(
            backend,
            flat_queries,
            snapshot,
            resolved_windows,
        ),
    )
    print("== build_flat_query_dim128 and flat-query tiled8 nested score_resolved_candidate_window_for_cpu ==")
    append_measurement(
        measurements,
        task,
        "build_flat_query_dim128_and_flat_query_tiled8_nested_score_resolved_candidate_window_for_cpu",
        candidate_k,
        mean_candidate_document_count,
        mean_candidate_vector_count,
        benchmark_build_flat_query_and_tiled8_mean_seconds(
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
        / "profile_document_proxy_stage2_flat_query_tiled4_compare_browsecomp_gold.tsv"
    )
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
