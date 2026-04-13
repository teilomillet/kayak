from std.collections import List

from kayak.collections.validation import (
    require_non_empty_string,
    require_non_negative_int,
)

from .planning_goal import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_EXACT_ONLY,
    SEARCH_PLANNING_GOAL_LATENCY_FIRST,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    require_search_planning_goal,
)
from .stage1_capabilities import (
    stage1_capabilities_for_candidate_generator_kind,
)


comptime SEARCH_PLANNER_STATUS_BENCHMARK_ONLY = "benchmark_only"
comptime SEARCH_PLANNER_STATUS_EXACT_FALLBACK = "exact_fallback"
comptime SEARCH_PLANNER_STATUS_EXPERIMENTAL = "experimental"
comptime SEARCH_PLANNER_STATUS_PROMOTED = "promoted"
comptime SEARCH_PLANNER_STATUS_REGRESSION_BASELINE = "regression_baseline"


struct SearchPlannerRegistryEntry(Copyable):
    var candidate_generator_kind: String
    var planner_status: String
    var balanced_priority: Int
    var latency_first_priority: Int
    var native_multivector_priority: Int

    def __init__(
        out self,
        var candidate_generator_kind: String,
        var planner_status: String,
        balanced_priority: Int,
        latency_first_priority: Int,
        native_multivector_priority: Int,
    ) raises:
        _ = stage1_capabilities_for_candidate_generator_kind(candidate_generator_kind)
        self.candidate_generator_kind = candidate_generator_kind^
        self.planner_status = require_non_empty_string(
            planner_status, "planner_status"
        )
        self.balanced_priority = require_non_negative_int(
            balanced_priority, "balanced_priority"
        )
        self.latency_first_priority = require_non_negative_int(
            latency_first_priority, "latency_first_priority"
        )
        self.native_multivector_priority = require_non_negative_int(
            native_multivector_priority, "native_multivector_priority"
        )


def registered_search_planner_entries() raises -> List[SearchPlannerRegistryEntry]:
    return [
        SearchPlannerRegistryEntry(
            "exact_full_scan",
            SEARCH_PLANNER_STATUS_EXACT_FALLBACK,
            700,
            700,
            700,
        ),
        SearchPlannerRegistryEntry(
            "document_proxy",
            SEARCH_PLANNER_STATUS_PROMOTED,
            100,
            100,
            600,
        ),
        SearchPlannerRegistryEntry(
            "centroid_postings_flat",
            SEARCH_PLANNER_STATUS_PROMOTED,
            200,
            200,
            200,
        ),
        SearchPlannerRegistryEntry(
            "centroid_postings_imputed_flat",
            SEARCH_PLANNER_STATUS_PROMOTED,
            300,
            500,
            100,
        ),
        SearchPlannerRegistryEntry(
            "centroid_postings",
            SEARCH_PLANNER_STATUS_REGRESSION_BASELINE,
            400,
            300,
            300,
        ),
        SearchPlannerRegistryEntry(
            "gem_graph",
            SEARCH_PLANNER_STATUS_EXPERIMENTAL,
            500,
            600,
            400,
        ),
        SearchPlannerRegistryEntry(
            "centroid_heads",
            SEARCH_PLANNER_STATUS_REGRESSION_BASELINE,
            600,
            400,
            500,
        ),
        SearchPlannerRegistryEntry(
            "centroid_postings_head",
            SEARCH_PLANNER_STATUS_BENCHMARK_ONLY,
            0,
            0,
            0,
        ),
        SearchPlannerRegistryEntry(
            "centroid_postings_head_auto",
            SEARCH_PLANNER_STATUS_BENCHMARK_ONLY,
            0,
            0,
            0,
        ),
        SearchPlannerRegistryEntry(
            "centroid_postings_blockmax",
            SEARCH_PLANNER_STATUS_BENCHMARK_ONLY,
            0,
            0,
            0,
        ),
        SearchPlannerRegistryEntry(
            "centroid_postings_imputed",
            SEARCH_PLANNER_STATUS_REGRESSION_BASELINE,
            0,
            0,
            0,
        ),
    ]


def search_planner_registry_entry(
    candidate_generator_kind: String
) raises -> SearchPlannerRegistryEntry:
    for entry in registered_search_planner_entries():
        if entry.candidate_generator_kind == candidate_generator_kind:
            return entry.copy()

    raise Error(
        "candidate generator kind is not registered for planner selection: "
        + candidate_generator_kind
    )


def planner_priority_for_goal(
    read entry: SearchPlannerRegistryEntry, goal: String
) raises -> Int:
    var normalized_goal = require_search_planning_goal(goal)
    if normalized_goal == SEARCH_PLANNING_GOAL_BALANCED:
        return entry.balanced_priority
    if normalized_goal == SEARCH_PLANNING_GOAL_LATENCY_FIRST:
        return entry.latency_first_priority
    if normalized_goal == SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR:
        return entry.native_multivector_priority
    if normalized_goal == SEARCH_PLANNING_GOAL_EXACT_ONLY:
        if entry.candidate_generator_kind == "exact_full_scan":
            return 1
        return 0

    raise Error("unknown search planning goal: " + normalized_goal)


def registered_search_planner_candidate_generator_kinds() raises -> List[String]:
    var kinds = List[String]()
    for entry in registered_search_planner_entries():
        kinds.append(entry.candidate_generator_kind.copy())
    return kinds^


def default_candidate_generator_order_for_goal(goal: String) raises -> List[String]:
    var normalized_goal = require_search_planning_goal(goal)
    if normalized_goal == SEARCH_PLANNING_GOAL_EXACT_ONLY:
        return ["exact_full_scan"]

    var ordered = List[String]()
    var ordered_priorities = List[Int]()

    for entry in registered_search_planner_entries():
        var priority = planner_priority_for_goal(entry, normalized_goal)
        if priority <= 0:
            continue

        var insert_at = len(ordered)
        for index in range(len(ordered_priorities)):
            if priority < ordered_priorities[index]:
                insert_at = index
                break

        ordered.insert(
            insert_at, entry.candidate_generator_kind.copy()
        )
        ordered_priorities.insert(insert_at, priority)

    return ordered^
