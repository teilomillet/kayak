# Explicit graph-frontier scheduling policy contract for graph-family stage 1.

from std.collections import List


comptime GRAPH_FRONTIER_POLICY_KIND_LOCAL_PER_ENTRY = "local_per_entry"
comptime GRAPH_FRONTIER_POLICY_KIND_GLOBAL_BEST_FIRST = "global_best_first"
comptime GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_PER_ENTRY = (
    "hybrid_best_head_per_entry"
)
comptime GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_FAIR_ROUND = (
    "hybrid_best_head_fair_round"
)
comptime GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_QUOTA2_ROUND = (
    "hybrid_best_head_quota2_round"
)
comptime DEFAULT_GRAPH_FRONTIER_POLICY_KIND = (
    GRAPH_FRONTIER_POLICY_KIND_LOCAL_PER_ENTRY
)


def graph_frontier_policy_kinds() -> List[String]:
    return [
        GRAPH_FRONTIER_POLICY_KIND_LOCAL_PER_ENTRY,
        GRAPH_FRONTIER_POLICY_KIND_GLOBAL_BEST_FIRST,
        GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_PER_ENTRY,
        GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_FAIR_ROUND,
        GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_QUOTA2_ROUND,
    ]


def require_graph_frontier_policy_kind(kind: String) raises -> String:
    for candidate_kind in graph_frontier_policy_kinds():
        if candidate_kind == kind:
            return kind.copy()

    raise Error("unknown graph_frontier_policy_kind: " + kind)
