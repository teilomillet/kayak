import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    default_public_benchmark_dataset_keys,
    ensure_public_benchmark_dataset_collection_mirror,
    load_public_benchmark_dataset,
    standard_candidate_window_sizes,
)
from kayak.collections import SnapshotId, load_resolved_collection_snapshot
from kayak.planning import (
    MutableCentroidCandidateGenerationWorkspace,
    candidate_generation_for_plan_with_workspace,
    explain_collection_search,
    best_effort_faithfulness_policy,
    centroid_postings_flat_search_plan,
    centroid_postings_imputed_flat_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import StoredJudgedTask


comptime CENTROID_HEAD_POSTING_CAP = 16
comptime BENCHMARK_QUERY_CYCLES = 64


struct WideWindowCandidateGenerationSummary(Copyable):
    var dataset_id: String
    var slice_name: String
    var candidate_generator_kind: String
    var final_k: Int
    var candidate_k: Int
    var query_count: Int
    var mean_candidate_generation_seconds: Float64
    var mean_candidate_hit_count: Float64
    var mean_candidate_recall_at_final_k: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var slice_name: String,
        var candidate_generator_kind: String,
        final_k: Int,
        candidate_k: Int,
        query_count: Int,
        mean_candidate_generation_seconds: Float64,
        mean_candidate_hit_count: Float64,
        mean_candidate_recall_at_final_k: Float64,
    ):
        self.dataset_id = dataset_id^
        self.slice_name = slice_name^
        self.candidate_generator_kind = candidate_generator_kind^
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_count = query_count
        self.mean_candidate_generation_seconds = mean_candidate_generation_seconds
        self.mean_candidate_hit_count = mean_candidate_hit_count
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k


def wide_window_candidate_budgets(final_k: Int, document_count: Int) -> List[Int]:
    var budgets = List[Int]()
    for candidate_k in standard_candidate_window_sizes(final_k, document_count):
        if candidate_k >= final_k * 8 or candidate_k == document_count:
            budgets.append(candidate_k)
    return budgets^


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_summary_json(
    mut buffer: String, read summary: WideWindowCandidateGenerationSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(summary.candidate_generator_kind) + "\","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"mean_candidate_generation_seconds\":"
    buffer += String(summary.mean_candidate_generation_seconds) + ","
    buffer += "\"mean_candidate_hit_count\":"
    buffer += String(summary.mean_candidate_hit_count) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k)
    buffer += "}"


def summaries_json(
    read summaries: List[WideWindowCandidateGenerationSummary]
) -> String:
    var buffer = "["
    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_summary_json(buffer, summaries[index])
    buffer += "]"
    return buffer


def build_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    snapshot_root: Path,
    candidate_k: Int,
    candidate_generator_kind: String,
) raises -> WideWindowCandidateGenerationSummary:
    var snapshot = load_resolved_collection_snapshot(
        snapshot_root,
        SnapshotId("snapshot-0001"),
    )
    var task = stored_task.task.copy()
    var plan = centroid_postings_imputed_flat_search_plan(
        task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
    )
    if candidate_generator_kind == "centroid_postings_flat":
        plan = centroid_postings_flat_search_plan(
            task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
        )

    var workspace = MutableCentroidCandidateGenerationWorkspace()
    var query_index = 0
    var benchmark_iters = len(task.queries) * BENCHMARK_QUERY_CYCLES

    def candidate_once() capturing raises:
        bench_compiler.keep(
            candidate_generation_for_plan_with_workspace(
                backend,
                task.queries[query_index].query,
                snapshot,
                plan,
                workspace,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[candidate_once](
        num_warmup_iters=len(task.queries),
        max_iters=benchmark_iters,
        min_runtime_secs=0.2,
        max_batch_size=1,
    )

    var hit_total = 0.0
    var recall_total = 0.0
    for judged_query in task.queries:
        var explain = explain_collection_search(
            backend,
            judged_query.query,
            snapshot,
            plan,
        )
        hit_total += Float64(len(explain.candidate_set.hits))
        recall_total += Float64(explain.candidate_recall_at_final_k)

    return WideWindowCandidateGenerationSummary(
        stored_task.dataset_id.copy(),
        task.slice_name.copy(),
        candidate_generator_kind.copy(),
        task.k,
        candidate_k,
        len(task.queries),
        Float64(report.mean()),
        hit_total / Float64(len(task.queries)),
        recall_total / Float64(len(task.queries)),
    )


def append_dataset_summaries(
    mut summaries: List[WideWindowCandidateGenerationSummary], dataset_key: String
) raises:
    var dataset = load_public_benchmark_dataset(dataset_key)
    var collection_root = ensure_public_benchmark_dataset_collection_mirror(
        dataset,
        "wide_window_candidate_generation",
        0,
        0,
        CENTROID_HEAD_POSTING_CAP,
        include_frontier_gem_graph=False,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var task = dataset.stored_task.task.copy()
    var backend = ExactCpuBackend()

    for candidate_k in wide_window_candidate_budgets(
        task.k,
        snapshot.snapshot.stats.document_count,
    ):
        summaries.append(
            build_summary(
                backend,
                dataset.stored_task,
                collection_root,
                candidate_k,
                "centroid_postings_flat",
            )
        )
        summaries.append(
            build_summary(
                backend,
                dataset.stored_task,
                collection_root,
                candidate_k,
                "centroid_postings_imputed_flat",
            )
        )


def main() raises:
    var summaries = List[WideWindowCandidateGenerationSummary]()
    for dataset_key in default_public_benchmark_dataset_keys():
        append_dataset_summaries(summaries, dataset_key)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "public_wide_window_candidate_generation.json"
    output_path.write_text(summaries_json(summaries))
    print("wrote ", String(output_path))
