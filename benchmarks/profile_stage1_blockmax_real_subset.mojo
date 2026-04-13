import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import (
    CollectionId,
    ExactCpuBackend,
    JudgedTask,
    NamespaceId,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SearchPlan,
    SnapshotId,
    StoredPackedIndex,
    TenantId,
    best_effort_faithfulness_policy,
    candidate_generation_for_plan,
    centroid_heads_search_plan,
    centroid_postings_flat_search_plan,
    centroid_posting_blockmax_scores_for_segment_profiled,
    centroid_postings_blockmax_search_plan,
    centroid_postings_head_auto_search_plan,
    centroid_postings_head_search_plan,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_search_plan,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_one_segment_collection_mirror,
    ensure_scifact_real_subset_cache,
    loaded_segment_has_search_artifact,
    loaded_segment_stored_centroid_postings_index,
    load_resolved_collection_snapshot,
)
from kayak.collections import ResolvedCollectionSnapshot


comptime CENTROID_HEAD_POSTING_CAP = 16


struct Stage1BlockmaxMeasurement(Copyable):
    var dataset_name: String
    var slice_name: String
    var candidate_generator_kind: String
    var candidate_k: Int
    var mean_stage1_seconds: Float64
    var mean_selected_centroid_count: Float64
    var mean_candidate_block_count: Float64
    var mean_visited_block_count: Float64
    var mean_skipped_block_count: Float64
    var mean_candidate_posting_count: Float64
    var mean_visited_posting_count: Float64
    var mean_skipped_posting_count: Float64

    def __init__(
        out self,
        var dataset_name: String,
        var slice_name: String,
        var candidate_generator_kind: String,
        candidate_k: Int,
        mean_stage1_seconds: Float64,
        mean_selected_centroid_count: Float64,
        mean_candidate_block_count: Float64,
        mean_visited_block_count: Float64,
        mean_skipped_block_count: Float64,
        mean_candidate_posting_count: Float64,
        mean_visited_posting_count: Float64,
        mean_skipped_posting_count: Float64,
    ):
        self.dataset_name = dataset_name^
        self.slice_name = slice_name^
        self.candidate_generator_kind = candidate_generator_kind^
        self.candidate_k = candidate_k
        self.mean_stage1_seconds = mean_stage1_seconds
        self.mean_selected_centroid_count = mean_selected_centroid_count
        self.mean_candidate_block_count = mean_candidate_block_count
        self.mean_visited_block_count = mean_visited_block_count
        self.mean_skipped_block_count = mean_skipped_block_count
        self.mean_candidate_posting_count = mean_candidate_posting_count
        self.mean_visited_posting_count = mean_visited_posting_count
        self.mean_skipped_posting_count = mean_skipped_posting_count


def stage1_candidate_k(task: JudgedTask, document_count: Int) -> Int:
    var candidate_k = task.k * 4
    if candidate_k > document_count:
        return document_count

    return candidate_k


def append_measurement_tsv_line(
    mut out: String, read measurement: Stage1BlockmaxMeasurement
):
    out += measurement.dataset_name
    out += "\t"
    out += measurement.slice_name
    out += "\t"
    out += measurement.candidate_generator_kind
    out += "\t"
    out += String(measurement.candidate_k)
    out += "\t"
    out += String(measurement.mean_stage1_seconds)
    out += "\t"
    out += String(measurement.mean_selected_centroid_count)
    out += "\t"
    out += String(measurement.mean_candidate_block_count)
    out += "\t"
    out += String(measurement.mean_visited_block_count)
    out += "\t"
    out += String(measurement.mean_skipped_block_count)
    out += "\t"
    out += String(measurement.mean_candidate_posting_count)
    out += "\t"
    out += String(measurement.mean_visited_posting_count)
    out += "\t"
    out += String(measurement.mean_skipped_posting_count)
    out += "\n"


def write_measurements_tsv(
    path: Path, measurements: List[Stage1BlockmaxMeasurement]
) raises:
    var lines = String()
    lines += "dataset_name\tslice_name\tcandidate_generator_kind\tcandidate_k\tmean_stage1_seconds\tmean_selected_centroid_count\tmean_candidate_block_count\tmean_visited_block_count\tmean_skipped_block_count\tmean_candidate_posting_count\tmean_visited_posting_count\tmean_skipped_posting_count\n"

    for measurement in measurements:
        append_measurement_tsv_line(lines, measurement)

    path.write_text(lines)


def benchmark_candidate_generation_mean_seconds(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> Float64:
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            candidate_generation_for_plan(
                backend,
                task.queries[query_index].query,
                snapshot,
                plan,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return report.mean()


def make_zero_profile_measurement(
    dataset_name: String,
    slice_name: String,
    candidate_generator_kind: String,
    candidate_k: Int,
    mean_stage1_seconds: Float64,
) -> Stage1BlockmaxMeasurement:
    return Stage1BlockmaxMeasurement(
        dataset_name.copy(),
        slice_name.copy(),
        candidate_generator_kind.copy(),
        candidate_k,
        mean_stage1_seconds,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def measure_blockmax_profile(
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    candidate_k: Int,
    mean_stage1_seconds: Float64,
) raises -> Stage1BlockmaxMeasurement:
    if len(snapshot.segments) != 1:
        raise Error("blockmax profile benchmark expects a one-segment snapshot")

    if not loaded_segment_has_search_artifact(
        snapshot.segments[0],
        SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    ):
        raise Error(
            "blockmax profile benchmark requires centroid postings sidecars"
        )

    var total_selected_centroid_count = Float64(0.0)
    var total_candidate_block_count = Float64(0.0)
    var total_visited_block_count = Float64(0.0)
    var total_skipped_block_count = Float64(0.0)
    var total_candidate_posting_count = Float64(0.0)
    var total_visited_posting_count = Float64(0.0)
    var total_skipped_posting_count = Float64(0.0)

    for judged_query in task.queries:
        var result = centroid_posting_blockmax_scores_for_segment_profiled(
            judged_query.query.token_vectors,
            loaded_segment_stored_centroid_postings_index(
                snapshot.segments[0]
            ).index,
            candidate_k,
        )
        total_selected_centroid_count += Float64(
            result.profile.selected_centroid_count
        )
        total_candidate_block_count += Float64(
            result.profile.candidate_block_count
        )
        total_visited_block_count += Float64(result.profile.visited_block_count)
        total_skipped_block_count += Float64(result.profile.skipped_block_count)
        total_candidate_posting_count += Float64(
            result.profile.candidate_posting_count
        )
        total_visited_posting_count += Float64(
            result.profile.visited_posting_count
        )
        total_skipped_posting_count += Float64(
            result.profile.skipped_posting_count
        )
    var query_count = Float64(len(task.queries))

    return Stage1BlockmaxMeasurement(
        dataset_name.copy(),
        task.slice_name.copy(),
        "centroid_postings_blockmax",
        candidate_k,
        mean_stage1_seconds,
        total_selected_centroid_count / query_count,
        total_candidate_block_count / query_count,
        total_visited_block_count / query_count,
        total_skipped_block_count / query_count,
        total_candidate_posting_count / query_count,
        total_visited_posting_count / query_count,
        total_skipped_posting_count / query_count,
    )


def append_plan_measurements(
    mut measurements: List[Stage1BlockmaxMeasurement],
    dataset_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    candidate_k: Int,
) raises:
    var backend = ExactCpuBackend()

    print(
        "dataset=",
        dataset_name,
        " slice=",
        task.slice_name,
        " candidate_k=",
        candidate_k,
    )
    print("")

    var centroid_postings_plan = centroid_postings_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_postings ==")
    measurements.append(
        make_zero_profile_measurement(
            dataset_name,
            task.slice_name,
            "centroid_postings",
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_postings_plan,
            ),
        )
    )

    var centroid_postings_flat_plan = centroid_postings_flat_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_postings_flat ==")
    measurements.append(
        make_zero_profile_measurement(
            dataset_name,
            task.slice_name,
            "centroid_postings_flat",
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_postings_flat_plan,
            ),
        )
    )

    var centroid_heads_plan = centroid_heads_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_heads ==")
    measurements.append(
        make_zero_profile_measurement(
            dataset_name,
            task.slice_name,
            "centroid_heads",
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_heads_plan,
            ),
        )
    )

    var centroid_head_plan = centroid_postings_head_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_postings_head ==")
    measurements.append(
        make_zero_profile_measurement(
            dataset_name,
            task.slice_name,
            "centroid_postings_head",
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_head_plan,
            ),
        )
    )

    var centroid_head_auto_plan = centroid_postings_head_auto_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_postings_head_auto ==")
    measurements.append(
        make_zero_profile_measurement(
            dataset_name,
            task.slice_name,
            "centroid_postings_head_auto",
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_head_auto_plan,
            ),
        )
    )

    var centroid_blockmax_plan = centroid_postings_blockmax_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_postings_blockmax ==")
    measurements.append(
        measure_blockmax_profile(
            dataset_name,
            task,
            snapshot,
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_blockmax_plan,
            ),
        )
    )

    var centroid_imputed_plan = centroid_postings_imputed_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_postings_imputed ==")
    measurements.append(
        make_zero_profile_measurement(
            dataset_name,
            task.slice_name,
            "centroid_postings_imputed",
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_imputed_plan,
            ),
        )
    )

    var centroid_imputed_flat_plan = centroid_postings_imputed_flat_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    print("== centroid_postings_imputed_flat ==")
    measurements.append(
        make_zero_profile_measurement(
            dataset_name,
            task.slice_name,
            "centroid_postings_imputed_flat",
            candidate_k,
            benchmark_candidate_generation_mean_seconds(
                backend,
                task,
                snapshot,
                centroid_imputed_flat_plan,
            ),
        )
    )


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


def main() raises:
    var measurements = List[Stage1BlockmaxMeasurement]()

    var scifact_cache = ensure_scifact_real_subset_cache()
    var scifact_snapshot = load_profile_snapshot(
        "scifact_real_subset_profile",
        Path(".cache/kayak/scifact_real_subset_stage1_profile"),
        scifact_cache.stored_index,
    )
    append_plan_measurements(
        measurements,
        "SciFact",
        scifact_cache.stored_task.task,
        scifact_snapshot,
        stage1_candidate_k(
            scifact_cache.stored_task.task,
            scifact_snapshot.snapshot.stats.document_count,
        ),
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    var fiqa_snapshot = load_profile_snapshot(
        "fiqa_real_subset_profile",
        Path(".cache/kayak/fiqa_real_subset_stage1_profile"),
        fiqa_cache.stored_index,
    )
    append_plan_measurements(
        measurements,
        "FIQA",
        fiqa_cache.stored_task.task,
        fiqa_snapshot,
        stage1_candidate_k(
            fiqa_cache.stored_task.task,
            fiqa_snapshot.snapshot.stats.document_count,
        ),
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    var limit_small_snapshot = load_profile_snapshot(
        "limit_small_real_subset_profile",
        Path(".cache/kayak/limit_small_real_subset_stage1_profile"),
        limit_small_cache.stored_index,
    )
    append_plan_measurements(
        measurements,
        "LIMIT-small",
        limit_small_cache.stored_task.task,
        limit_small_snapshot,
        stage1_candidate_k(
            limit_small_cache.stored_task.task,
            limit_small_snapshot.snapshot.stats.document_count,
        ),
    )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "profile_stage1_blockmax_real_subset.tsv"
    write_measurements_tsv(output_path, measurements)
    print("wrote ", String(output_path))
