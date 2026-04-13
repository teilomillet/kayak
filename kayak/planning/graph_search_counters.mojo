# Graph-native counters for stage-1 traversal.
#
# Owns:
# - query-time counters that only make sense for graph-family candidate stages
#
# Does not own:
# - generic artifact shape counters such as tokens, vectors, or bytes


struct GraphSearchCounters(Copyable):
    var visited_vertex_count: Int
    var expanded_edge_count: Int
    var visited_cluster_count: Int
    var entry_point_count: Int
    var max_frontier_size: Int

    def __init__(out self):
        self.visited_vertex_count = 0
        self.expanded_edge_count = 0
        self.visited_cluster_count = 0
        self.entry_point_count = 0
        self.max_frontier_size = 0

    def __init__(
        out self,
        visited_vertex_count: Int,
        expanded_edge_count: Int,
        visited_cluster_count: Int,
        entry_point_count: Int,
        max_frontier_size: Int,
    ) raises:
        if visited_vertex_count < 0:
            raise Error(
                "graph search visited_vertex_count must be non-negative"
            )
        if expanded_edge_count < 0:
            raise Error("graph search expanded_edge_count must be non-negative")
        if visited_cluster_count < 0:
            raise Error("graph search visited_cluster_count must be non-negative")
        if entry_point_count < 0:
            raise Error("graph search entry_point_count must be non-negative")
        if max_frontier_size < 0:
            raise Error("graph search max_frontier_size must be non-negative")

        self.visited_vertex_count = visited_vertex_count
        self.expanded_edge_count = expanded_edge_count
        self.visited_cluster_count = visited_cluster_count
        self.entry_point_count = entry_point_count
        self.max_frontier_size = max_frontier_size

