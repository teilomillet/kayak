comptime SEARCH_PLANNING_GOAL_BALANCED = "balanced"
comptime SEARCH_PLANNING_GOAL_EXACT_ONLY = "exact_only"
comptime SEARCH_PLANNING_GOAL_LATENCY_FIRST = "latency_first"
comptime SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR = "native_multivector"


def require_search_planning_goal(goal: String) raises -> String:
    if goal == SEARCH_PLANNING_GOAL_BALANCED:
        return goal.copy()
    if goal == SEARCH_PLANNING_GOAL_EXACT_ONLY:
        return goal.copy()
    if goal == SEARCH_PLANNING_GOAL_LATENCY_FIRST:
        return goal.copy()
    if goal == SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR:
        return goal.copy()

    raise Error("unknown search planning goal: " + goal)
