from std.collections import List

from kayak.collections import (
    ResolvedCollectionSnapshot,
    SnapshotSearchArtifactAvailability,
)
from kayak.planning import (
    SearchPlanSelectionRequest,
    Stage2Operator,
    search_plan_with_stage2_operator,
    select_search_plan_for_availability,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import StoredJudgedTask

from .faithfulness_frontier_json import (
    FaithfulnessFrontierSummary,
    build_faithfulness_frontier_summary_for_plan,
    faithfulness_frontier_summary_json,
)
from .json_common import json_escape


struct PlannerBenchmarkSummary(Copyable):
    var planning_goal: String
    var stage2_kind: String
    var selected_candidate_generator_kind: String
    var selected_candidate_generator_status: String
    var selection_reason: String
    var available_candidate_generator_kinds: List[String]
    var effective_candidate_generator_order: List[String]
    var measured: FaithfulnessFrontierSummary

    def __init__(
        out self,
        var planning_goal: String,
        var stage2_kind: String,
        var selected_candidate_generator_kind: String,
        var selected_candidate_generator_status: String,
        var selection_reason: String,
        read available_candidate_generator_kinds: List[String],
        read effective_candidate_generator_order: List[String],
        measured: FaithfulnessFrontierSummary,
    ):
        self.planning_goal = planning_goal^
        self.stage2_kind = stage2_kind^
        self.selected_candidate_generator_kind = (
            selected_candidate_generator_kind^
        )
        self.selected_candidate_generator_status = (
            selected_candidate_generator_status^
        )
        self.selection_reason = selection_reason^
        self.available_candidate_generator_kinds = (
            available_candidate_generator_kinds.copy()
        )
        self.effective_candidate_generator_order = (
            effective_candidate_generator_order.copy()
        )
        self.measured = measured.copy()


def build_planner_benchmark_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read availability: SnapshotSearchArtifactAvailability,
    read request: SearchPlanSelectionRequest,
    stage2_operator: Stage2Operator,
    query_vector_budget: Int,
    requested_stage1_vector_budget: Int = 0,
    posting_cap: Int = 0,
) raises -> PlannerBenchmarkSummary:
    var selection = select_search_plan_for_availability(availability, request)
    var plan = search_plan_with_stage2_operator(selection.plan, stage2_operator)
    return PlannerBenchmarkSummary(
        request.goal.copy(),
        stage2_operator.kind.copy(),
        selection.plan.candidate_generator.kind.copy(),
        selection.selected_candidate_generator_status.copy(),
        selection.reason.copy(),
        selection.available_candidate_generator_kinds,
        selection.effective_candidate_generator_order,
        build_faithfulness_frontier_summary_for_plan(
            backend,
            stored_task,
            snapshot,
            plan,
            query_vector_budget,
            requested_stage1_vector_budget,
            posting_cap,
        ),
    )


def append_planner_benchmark_summary_json(
    mut buffer: String, read summary: PlannerBenchmarkSummary
):
    buffer += "{"
    buffer += "\"planning_goal\":\"" + json_escape(summary.planning_goal) + "\","
    buffer += "\"stage2_kind\":\"" + json_escape(summary.stage2_kind) + "\","
    buffer += "\"selected_candidate_generator_kind\":\""
    buffer += json_escape(summary.selected_candidate_generator_kind) + "\","
    buffer += "\"selected_candidate_generator_status\":\""
    buffer += json_escape(summary.selected_candidate_generator_status) + "\","
    buffer += "\"selection_reason\":\"" + json_escape(summary.selection_reason) + "\","
    buffer += "\"available_candidate_generator_kinds\":["
    for index in range(len(summary.available_candidate_generator_kinds)):
        if index > 0:
            buffer += ","
        buffer += "\""
        buffer += json_escape(summary.available_candidate_generator_kinds[index])
        buffer += "\""
    buffer += "],"
    buffer += "\"effective_candidate_generator_order\":["
    for index in range(len(summary.effective_candidate_generator_order)):
        if index > 0:
            buffer += ","
        buffer += "\""
        buffer += json_escape(summary.effective_candidate_generator_order[index])
        buffer += "\""
    buffer += "],"
    buffer += "\"measured\":"
    buffer += faithfulness_frontier_summary_json(summary.measured)
    buffer += "}"


def planner_benchmark_summary_json(read summary: PlannerBenchmarkSummary) -> String:
    var buffer = String()
    append_planner_benchmark_summary_json(buffer, summary)
    return buffer^


def planner_benchmark_summaries_json(
    read summaries: List[PlannerBenchmarkSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_planner_benchmark_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
