from std.collections import List

from kayak.collections import (
    ResolvedCollectionSnapshot,
    SnapshotSearchArtifactAvailability,
)
from kayak.planning import (
    SearchPlanSelectionRequest,
    SearchPlannerRegistryEntry,
    SearchPlan,
    Stage2ReferenceOperator,
    Stage3VerifierOperator,
    explain_collection_search,
    search_plan_for_candidate_generator_kind,
    search_planner_registry_entry,
    select_search_plan_for_availability,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import StoredJudgedTask

from .faithfulness_frontier_json import (
    FaithfulnessFrontierSummary,
    build_faithfulness_frontier_summary_for_plan,
    faithfulness_frontier_summary_json,
)
from .json_common import append_json_string_list, json_escape
from .materialized_artifact_families import materialized_artifact_families
from .query_text_support import judged_query_text_for_plan
from .search_plan_semantics_json import append_search_plan_semantics_json_fields


struct PlannerEvidenceCandidateSummary(Copyable):
    var candidate_generator_kind: String
    var planner_status: String
    var is_selected: Bool
    var measured: FaithfulnessFrontierSummary

    def __init__(
        out self,
        var candidate_generator_kind: String,
        var planner_status: String,
        is_selected: Bool,
        measured: FaithfulnessFrontierSummary,
    ):
        self.candidate_generator_kind = candidate_generator_kind^
        self.planner_status = planner_status^
        self.is_selected = is_selected
        self.measured = measured.copy()


struct PlannerEvidenceSummary(Copyable):
    var planning_goal: String
    var plan: SearchPlan
    var stage2_reference_materialized_artifact_families: List[String]
    var stage3_verifier_materialized_artifact_families: List[String]
    var selected_candidate_generator_status: String
    var selection_reason: String
    var available_candidate_generator_kinds: List[String]
    var effective_candidate_generator_order: List[String]
    var selected_is_undominated: Bool
    var dominating_candidate_generator_kind: String
    var dominance_reason: String
    var candidates: List[PlannerEvidenceCandidateSummary]

    def __init__(
        out self,
        var planning_goal: String,
        plan: SearchPlan,
        read stage2_reference_materialized_artifact_families: List[String],
        read stage3_verifier_materialized_artifact_families: List[String],
        var selected_candidate_generator_status: String,
        var selection_reason: String,
        read available_candidate_generator_kinds: List[String],
        read effective_candidate_generator_order: List[String],
        selected_is_undominated: Bool,
        var dominating_candidate_generator_kind: String,
        var dominance_reason: String,
        read candidates: List[PlannerEvidenceCandidateSummary],
    ):
        self.planning_goal = planning_goal^
        self.plan = plan.copy()
        self.stage2_reference_materialized_artifact_families = (
            stage2_reference_materialized_artifact_families.copy()
        )
        self.stage3_verifier_materialized_artifact_families = (
            stage3_verifier_materialized_artifact_families.copy()
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
        self.selected_is_undominated = selected_is_undominated
        self.dominating_candidate_generator_kind = (
            dominating_candidate_generator_kind^
        )
        self.dominance_reason = dominance_reason^
        self.candidates = candidates.copy()


def candidate_summary_dominates(
    read left: PlannerEvidenceCandidateSummary,
    read right: PlannerEvidenceCandidateSummary,
) -> Bool:
    var no_worse = (
        left.measured.mean_ndcg_at_k >= right.measured.mean_ndcg_at_k
        and left.measured.mean_recall_at_k >= right.measured.mean_recall_at_k
        and left.measured.mean_candidate_recall_at_final_k
        >= right.measured.mean_candidate_recall_at_final_k
        and left.measured.mean_search_seconds
        <= right.measured.mean_search_seconds
    )
    var strictly_better = (
        left.measured.mean_ndcg_at_k > right.measured.mean_ndcg_at_k
        or left.measured.mean_recall_at_k > right.measured.mean_recall_at_k
        or left.measured.mean_candidate_recall_at_final_k
        > right.measured.mean_candidate_recall_at_final_k
        or left.measured.mean_search_seconds
        < right.measured.mean_search_seconds
    )
    return no_worse and strictly_better


def candidate_summary_for_kind(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read request: SearchPlanSelectionRequest,
    read stage2_reference_operator: Stage2ReferenceOperator,
    read stage3_verifier: Stage3VerifierOperator,
    candidate_generator_kind: String,
    query_vector_budget: Int,
    requested_stage1_vector_budget: Int,
    posting_cap: Int,
    selected_candidate_generator_kind: String,
) raises -> PlannerEvidenceCandidateSummary:
    var registry_entry: SearchPlannerRegistryEntry = search_planner_registry_entry(
        candidate_generator_kind
    )
    var selected_plan = search_plan_for_candidate_generator_kind(
        candidate_generator_kind,
        request,
    )
    var plan = SearchPlan(
        selected_plan.candidate_generator,
        selected_plan.candidate_budget,
        selected_plan.faithfulness_policy,
        selected_plan.reference_scoring_semantics,
        stage2_reference_operator,
        stage3_verifier,
    )
    return PlannerEvidenceCandidateSummary(
        candidate_generator_kind.copy(),
        registry_entry.planner_status.copy(),
        candidate_generator_kind == selected_candidate_generator_kind,
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


def build_planner_evidence_summary(
    read backend: ExactCpuBackend,
    read stored_task: StoredJudgedTask,
    read snapshot: ResolvedCollectionSnapshot,
    read availability: SnapshotSearchArtifactAvailability,
    read request: SearchPlanSelectionRequest,
    read stage2_reference_operator: Stage2ReferenceOperator,
    read stage3_verifier: Stage3VerifierOperator,
    query_vector_budget: Int,
    requested_stage1_vector_budget: Int = 0,
    posting_cap: Int = 0,
) raises -> PlannerEvidenceSummary:
    var selection = select_search_plan_for_availability(availability, request)
    var selected_plan = SearchPlan(
        selection.plan.candidate_generator,
        selection.plan.candidate_budget,
        selection.plan.faithfulness_policy,
        selection.plan.reference_scoring_semantics,
        stage2_reference_operator,
        stage3_verifier,
    )
    var candidates = List[PlannerEvidenceCandidateSummary]()

    for candidate_generator_kind in selection.available_candidate_generator_kinds:
        candidates.append(
            candidate_summary_for_kind(
                backend,
                stored_task,
                snapshot,
                request,
                stage2_reference_operator,
                stage3_verifier,
                candidate_generator_kind,
                query_vector_budget,
                requested_stage1_vector_budget,
                posting_cap,
                selected_plan.candidate_generator.kind,
            )
        )

    var selected_index = -1
    for index in range(len(candidates)):
        if candidates[index].is_selected:
            selected_index = index
            break

    if selected_index < 0:
        raise Error("planner evidence summary could not locate the selected candidate")

    var dominating_candidate_generator_kind = String()
    var dominance_reason = String("selected candidate is undominated on current evidence")

    for index in range(len(candidates)):
        if index == selected_index:
            continue
        if candidate_summary_dominates(candidates[index], candidates[selected_index]):
            dominating_candidate_generator_kind = (
                candidates[index].candidate_generator_kind.copy()
            )
            dominance_reason = (
                "candidate "
                + dominating_candidate_generator_kind
                + " dominates the selected generator on ndcg, recall, candidate recall, and latency"
            )
            break

    var representative_query = stored_task.task.queries[0].copy()
    var representative_explain = explain_collection_search(
        backend,
        representative_query.query,
        snapshot,
        selected_plan,
        query_text=judged_query_text_for_plan(
            selected_plan,
            representative_query.description,
        ),
    )

    return PlannerEvidenceSummary(
        request.goal.copy(),
        selected_plan,
        materialized_artifact_families(
            representative_explain.stage2.materialized_artifacts
        ),
        materialized_artifact_families(
            representative_explain.stage3_verifier.materialized_artifacts
        ),
        selection.selected_candidate_generator_status.copy(),
        selection.reason.copy(),
        selection.available_candidate_generator_kinds,
        selection.effective_candidate_generator_order,
        dominating_candidate_generator_kind.byte_length() == 0,
        dominating_candidate_generator_kind^,
        dominance_reason^,
        candidates,
    )


def append_planner_evidence_candidate_summary_json(
    mut buffer: String,
    read summary: PlannerEvidenceCandidateSummary,
):
    buffer += "{"
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(summary.candidate_generator_kind) + "\","
    buffer += "\"planner_status\":\""
    buffer += json_escape(summary.planner_status) + "\","
    buffer += "\"is_selected\":"
    if summary.is_selected:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"measured\":"
    buffer += faithfulness_frontier_summary_json(summary.measured)
    buffer += "}"


def append_planner_evidence_summary_json(
    mut buffer: String,
    read summary: PlannerEvidenceSummary,
):
    buffer += "{"
    buffer += "\"planning_goal\":\"" + json_escape(summary.planning_goal) + "\","
    append_search_plan_semantics_json_fields(buffer, summary.plan)
    buffer += ","
    buffer += "\"selected_candidate_generator_kind\":\""
    buffer += json_escape(summary.plan.candidate_generator.kind) + "\","
    buffer += "\"stage2_reference_materialized_artifact_families\":"
    append_json_string_list(
        buffer,
        summary.stage2_reference_materialized_artifact_families,
    )
    buffer += ","
    buffer += "\"stage3_verifier_materialized_artifact_families\":"
    append_json_string_list(
        buffer,
        summary.stage3_verifier_materialized_artifact_families,
    )
    buffer += ","
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
    buffer += "\"selected_is_undominated\":"
    if summary.selected_is_undominated:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"dominating_candidate_generator_kind\":\""
    buffer += json_escape(summary.dominating_candidate_generator_kind) + "\","
    buffer += "\"dominance_reason\":\""
    buffer += json_escape(summary.dominance_reason) + "\","
    buffer += "\"candidates\":["
    for index in range(len(summary.candidates)):
        if index > 0:
            buffer += ","
        append_planner_evidence_candidate_summary_json(
            buffer,
            summary.candidates[index],
        )
    buffer += "]"
    buffer += "}"


def planner_evidence_summary_json(read summary: PlannerEvidenceSummary) -> String:
    var buffer = String()
    append_planner_evidence_summary_json(buffer, summary)
    return buffer^


def planner_evidence_summaries_json(
    read summaries: List[PlannerEvidenceSummary]
) -> String:
    var buffer = String()
    buffer += "["
    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_planner_evidence_summary_json(buffer, summaries[index])
    buffer += "]"
    return buffer^
