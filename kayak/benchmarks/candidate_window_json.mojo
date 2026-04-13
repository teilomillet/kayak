from std.benchmark import run
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.eval import JudgedTask
from kayak.planning import (
    CandidateGenerator,
    SearchPlan,
    candidate_generation_for_plan,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
    explain_collection_search,
)
from kayak.runtime import ExactCpuBackend

from .query_text_support import judged_query_text_for_plan
from .search_plan_semantics_json import (
    append_candidate_generator_semantics_json_fields,
)


struct CandidateWindowSweepSummary(Copyable):
    var dataset_id: String
    var collection_id: String
    var snapshot_id: String
    var model_name: String
    var candidate_generator: CandidateGenerator
    var final_k: Int
    var candidate_k: Int
    var query_vector_budget: Int
    var document_vector_budget: Int
    var query_count: Int
    var mean_candidate_generation_seconds: Float64
    var mean_candidate_hit_count: Float64
    var mean_candidate_recall_at_final_k: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var collection_id: String,
        var snapshot_id: String,
        var model_name: String,
        candidate_generator: CandidateGenerator,
        final_k: Int,
        candidate_k: Int,
        query_vector_budget: Int,
        document_vector_budget: Int,
        query_count: Int,
        mean_candidate_generation_seconds: Float64,
        mean_candidate_hit_count: Float64,
        mean_candidate_recall_at_final_k: Float64,
    ):
        self.dataset_id = dataset_id^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.model_name = model_name^
        self.candidate_generator = candidate_generator.copy()
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_vector_budget = query_vector_budget
        self.document_vector_budget = document_vector_budget
        self.query_count = query_count
        self.mean_candidate_generation_seconds = mean_candidate_generation_seconds
        self.mean_candidate_hit_count = mean_candidate_hit_count
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k


def standard_candidate_window_sizes(
    final_k: Int, document_count: Int
) -> List[Int]:
    var sizes = List[Int]()
    var candidate_size = final_k
    if candidate_size > document_count:
        candidate_size = document_count
    sizes.append(candidate_size)

    candidate_size = final_k * 2
    if candidate_size > document_count:
        candidate_size = document_count
    if sizes[len(sizes) - 1] != candidate_size:
        sizes.append(candidate_size)

    candidate_size = final_k * 4
    if candidate_size > document_count:
        candidate_size = document_count
    if sizes[len(sizes) - 1] != candidate_size:
        sizes.append(candidate_size)

    candidate_size = final_k * 8
    if candidate_size > document_count:
        candidate_size = document_count
    if sizes[len(sizes) - 1] != candidate_size:
        sizes.append(candidate_size)

    candidate_size = document_count
    if sizes[len(sizes) - 1] != candidate_size:
        sizes.append(candidate_size)
    return sizes^


def build_candidate_window_sweep_summary(
    read backend: ExactCpuBackend,
    dataset_id: String,
    model_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    candidate_k: Int,
) raises -> CandidateWindowSweepSummary:
    return build_candidate_window_sweep_summary_for_plan(
        backend,
        dataset_id,
        model_name,
        task,
        snapshot,
        exact_full_scan_search_plan(task.k, candidate_k),
        0,
        0,
    )


def build_candidate_window_sweep_summary_for_plan(
    read backend: ExactCpuBackend,
    dataset_id: String,
    model_name: String,
    read task: JudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    query_vector_budget: Int,
    document_vector_budget: Int,
) raises -> CandidateWindowSweepSummary:
    var candidate_hit_total = 0.0
    var recall_total = 0.0
    var query_index = 0

    def candidate_once() capturing raises:
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

    var candidate_report = run[candidate_once](
        num_warmup_iters=0,
        max_iters=len(task.queries),
        min_runtime_secs=0.0,
        max_batch_size=1,
    )

    for judged_query in task.queries:
        var explain = explain_collection_search(
            backend,
            judged_query.query,
            snapshot,
            plan,
            query_text=judged_query_text_for_plan(
                plan,
                judged_query.description,
            ),
        )
        candidate_hit_total += Float64(len(explain.candidate_set.hits))
        recall_total += Float64(explain.candidate_recall_at_final_k)

    return CandidateWindowSweepSummary(
        dataset_id,
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        model_name,
        plan.candidate_generator,
        task.k,
        plan.candidate_budget.candidate_k,
        query_vector_budget,
        document_vector_budget,
        len(task.queries),
        Float64(candidate_report.mean()),
        candidate_hit_total / Float64(len(task.queries)),
        recall_total / Float64(len(task.queries)),
    )


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_candidate_window_sweep_summary_json(
    mut buffer: String, read summary: CandidateWindowSweepSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    append_candidate_generator_semantics_json_fields(
        buffer,
        summary.candidate_generator,
    )
    buffer += ","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_vector_budget\":" + String(summary.query_vector_budget) + ","
    buffer += "\"document_vector_budget\":"
    buffer += String(summary.document_vector_budget) + ","
    buffer += "\"query_count\":" + String(summary.query_count) + ","
    buffer += "\"mean_candidate_generation_seconds\":"
    buffer += String(summary.mean_candidate_generation_seconds) + ","
    buffer += "\"mean_candidate_hit_count\":"
    buffer += String(summary.mean_candidate_hit_count) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k)
    buffer += "}"


def candidate_window_sweep_summaries_json(
    read summaries: List[CandidateWindowSweepSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_candidate_window_sweep_summary_json(buffer, summaries[index])
    buffer += "]"
    return buffer^
