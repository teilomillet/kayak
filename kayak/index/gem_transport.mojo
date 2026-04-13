from std.collections import List

from kayak.numeric import MetricScalar, zero_metric_scalar


comptime TRANSPORT_FLOW_EPSILON = MetricScalar(1.0e-12)


def max_metric_scalar() -> MetricScalar:
    return MetricScalar(1.0e300)


struct TransportGraph(Copyable):
    var adjacency: List[List[Int]]
    var to_nodes: List[Int]
    var reverse_indices: List[Int]
    var capacities: List[MetricScalar]
    var costs: List[MetricScalar]

    def __init__(out self, node_count: Int):
        self.adjacency = List[List[Int]]()
        for _ in range(node_count):
            self.adjacency.append(List[Int]())
        self.to_nodes = List[Int]()
        self.reverse_indices = List[Int]()
        self.capacities = List[MetricScalar]()
        self.costs = List[MetricScalar]()


def add_transport_edge(
    mut graph: TransportGraph,
    from_node: Int,
    to_node: Int,
    capacity: MetricScalar,
    cost: MetricScalar,
) raises:
    if capacity < zero_metric_scalar():
        raise Error("transport edge capacity must be non-negative")
    if from_node < 0 or from_node >= len(graph.adjacency):
        raise Error("transport edge from_node is out of range")
    if to_node < 0 or to_node >= len(graph.adjacency):
        raise Error("transport edge to_node is out of range")

    var forward_index = len(graph.to_nodes)
    var reverse_index = forward_index + 1

    graph.to_nodes.append(to_node)
    graph.reverse_indices.append(reverse_index)
    graph.capacities.append(capacity)
    graph.costs.append(cost)
    graph.adjacency[from_node].append(forward_index)

    graph.to_nodes.append(from_node)
    graph.reverse_indices.append(forward_index)
    graph.capacities.append(zero_metric_scalar())
    graph.costs.append(zero_metric_scalar() - cost)
    graph.adjacency[to_node].append(reverse_index)


def min_cost_transport_distance(
    read left_masses: List[MetricScalar],
    read right_masses: List[MetricScalar],
    read pair_costs: List[MetricScalar],
) raises -> MetricScalar:
    if len(left_masses) == 0 or len(right_masses) == 0:
        raise Error("transport requires non-empty left and right masses")
    if len(pair_costs) != len(left_masses) * len(right_masses):
        raise Error("transport pair_costs must match left_count * right_count")

    var total_left = zero_metric_scalar()
    for mass in left_masses:
        if mass < zero_metric_scalar():
            raise Error("transport left masses must be non-negative")
        total_left += mass
    var total_right = zero_metric_scalar()
    for mass in right_masses:
        if mass < zero_metric_scalar():
            raise Error("transport right masses must be non-negative")
        total_right += mass

    if total_left <= TRANSPORT_FLOW_EPSILON or total_right <= TRANSPORT_FLOW_EPSILON:
        raise Error("transport requires positive total mass")

    var node_count = 2 + len(left_masses) + len(right_masses)
    var source = 0
    var sink = node_count - 1
    var graph = TransportGraph(node_count)

    for left_index in range(len(left_masses)):
        add_transport_edge(
            graph,
            source,
            1 + left_index,
            left_masses[left_index],
            zero_metric_scalar(),
        )
    for right_index in range(len(right_masses)):
        add_transport_edge(
            graph,
            1 + len(left_masses) + right_index,
            sink,
            right_masses[right_index],
            zero_metric_scalar(),
        )
    for left_index in range(len(left_masses)):
        for right_index in range(len(right_masses)):
            add_transport_edge(
                graph,
                1 + left_index,
                1 + len(left_masses) + right_index,
                MetricScalar(1.0),
                pair_costs[left_index * len(right_masses) + right_index],
            )

    var potentials = List[MetricScalar]()
    var distances = List[MetricScalar]()
    var previous_nodes = List[Int]()
    var previous_edges = List[Int]()
    var visited = List[Bool]()
    for _ in range(node_count):
        potentials.append(zero_metric_scalar())
        distances.append(max_metric_scalar())
        previous_nodes.append(-1)
        previous_edges.append(-1)
        visited.append(False)

    var remaining_flow = total_left
    var total_cost = zero_metric_scalar()
    while remaining_flow > TRANSPORT_FLOW_EPSILON:
        for node_index in range(node_count):
            distances[node_index] = max_metric_scalar()
            previous_nodes[node_index] = -1
            previous_edges[node_index] = -1
            visited[node_index] = False
        distances[source] = zero_metric_scalar()

        while True:
            var current = -1
            var current_distance = max_metric_scalar()
            for node_index in range(node_count):
                if visited[node_index]:
                    continue
                if distances[node_index] < current_distance:
                    current_distance = distances[node_index]
                    current = node_index

            if current == -1:
                break

            visited[current] = True
            for edge_index in graph.adjacency[current]:
                if graph.capacities[edge_index] <= TRANSPORT_FLOW_EPSILON:
                    continue
                var next_node = graph.to_nodes[edge_index]
                var reduced_cost = (
                    distances[current]
                    + graph.costs[edge_index]
                    + potentials[current]
                    - potentials[next_node]
                )
                if reduced_cost < distances[next_node]:
                    distances[next_node] = reduced_cost
                    previous_nodes[next_node] = current
                    previous_edges[next_node] = edge_index

        if previous_nodes[sink] == -1:
            raise Error("transport graph could not route the full mass")

        for node_index in range(node_count):
            if distances[node_index] < max_metric_scalar():
                potentials[node_index] += distances[node_index]

        var augment = remaining_flow
        var node = sink
        while node != source:
            var edge_index = previous_edges[node]
            if edge_index == -1:
                raise Error("transport path reconstruction failed")
            if graph.capacities[edge_index] < augment:
                augment = graph.capacities[edge_index]
            node = previous_nodes[node]

        node = sink
        while node != source:
            var edge_index = previous_edges[node]
            graph.capacities[edge_index] -= augment
            var reverse_index = graph.reverse_indices[edge_index]
            graph.capacities[reverse_index] += augment
            node = previous_nodes[node]

        total_cost += augment * potentials[sink]
        remaining_flow -= augment

    return total_cost
