from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    PlannerEvidenceSummary,
    build_planner_evidence_summary,
    ensure_public_benchmark_dataset_collection_mirror,
    load_public_benchmark_dataset,
    planner_evidence_summaries_json,
    require_public_benchmark_dataset_loaded_text_corpus,
    standard_candidate_window_sizes,
)
from kayak.collections import (
    SnapshotId,
    load_resolved_collection_snapshot,
    load_snapshot_search_artifact_availability,
)
from kayak.eval import JudgedTask
from kayak.planning import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_EXACT_ONLY,
    SEARCH_PLANNING_GOAL_LATENCY_FIRST,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    SearchPlanSelectionRequest,
    best_effort_faithfulness_policy,
    clause_text_stage2_operator,
    exact_late_interaction_stage2_operator,
)
from kayak.runtime import ExactCpuBackend


comptime CENTROID_HEAD_POSTING_CAP = 16


def max_query_vector_budget(read task: JudgedTask) -> Int:
    var max_budget = 0
    for judged_query in task.queries:
        if judged_query.query.vector_count > max_budget:
            max_budget = judged_query.query.vector_count

    if max_budget <= 0:
        return 1
    return max_budget


def main() raises:
    var dataset = load_public_benchmark_dataset(
        "browsecomp_plus_gold",
        load_text_corpus=True,
    )
    var task = dataset.stored_task.task.copy()
    _ = require_public_benchmark_dataset_loaded_text_corpus(dataset)
    var collection_root = ensure_public_benchmark_dataset_collection_mirror(
        dataset,
        "planner_evidence_collection",
        0,
        0,
        CENTROID_HEAD_POSTING_CAP,
        include_frontier_gem_graph=True,
    )
    var snapshot = load_resolved_collection_snapshot(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var availability = load_snapshot_search_artifact_availability(
        collection_root,
        SnapshotId("snapshot-0001"),
    )
    var summaries = List[PlannerEvidenceSummary]()
    var backend = ExactCpuBackend()
    var query_budget = max_query_vector_budget(task)
    var stage2_operators = [
        exact_late_interaction_stage2_operator(),
        clause_text_stage2_operator(),
    ]

    for candidate_k in standard_candidate_window_sizes(
        task.k,
        snapshot.snapshot.stats.document_count,
    ):
        for stage2_operator in stage2_operators:
            summaries.append(
                build_planner_evidence_summary(
                    backend,
                    dataset.stored_task,
                    snapshot,
                    availability,
                    SearchPlanSelectionRequest(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                        goal=SEARCH_PLANNING_GOAL_BALANCED,
                    ),
                    stage2_operator,
                    query_budget,
                    0,
                    CENTROID_HEAD_POSTING_CAP,
                )
            )
            summaries.append(
                build_planner_evidence_summary(
                    backend,
                    dataset.stored_task,
                    snapshot,
                    availability,
                    SearchPlanSelectionRequest(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                        goal=SEARCH_PLANNING_GOAL_LATENCY_FIRST,
                    ),
                    stage2_operator,
                    query_budget,
                    0,
                    CENTROID_HEAD_POSTING_CAP,
                )
            )
            summaries.append(
                build_planner_evidence_summary(
                    backend,
                    dataset.stored_task,
                    snapshot,
                    availability,
                    SearchPlanSelectionRequest(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                        goal=SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
                    ),
                    stage2_operator,
                    query_budget,
                    0,
                    CENTROID_HEAD_POSTING_CAP,
                )
            )
            summaries.append(
                build_planner_evidence_summary(
                    backend,
                    dataset.stored_task,
                    snapshot,
                    availability,
                    SearchPlanSelectionRequest(
                        task.k,
                        candidate_k,
                        best_effort_faithfulness_policy(),
                        goal=SEARCH_PLANNING_GOAL_EXACT_ONLY,
                    ),
                    stage2_operator,
                    query_budget,
                    0,
                    CENTROID_HEAD_POSTING_CAP,
                )
            )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "browsecomp_plus_gold_planner_evidence.json"
    output_path.write_text(planner_evidence_summaries_json(summaries))
    print("wrote ", String(output_path))
