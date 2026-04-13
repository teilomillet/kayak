from std.collections import List

from kayak.numeric import MetricScalar, zero_metric_scalar


def one_metric_scalar() -> MetricScalar:
    return MetricScalar(1.0)


def max_metric_scalar() -> MetricScalar:
    return MetricScalar(1.0e300)


struct AdaptiveCutoffDecisionTree(Copyable):
    var split_feature_indices: List[Int]
    var split_thresholds: List[MetricScalar]
    var predicted_labels: List[Int]
    var left_children: List[Int]
    var right_children: List[Int]
    var feature_count: Int

    def __init__(out self):
        self.split_feature_indices = List[Int]()
        self.split_thresholds = List[MetricScalar]()
        self.predicted_labels = List[Int]()
        self.left_children = List[Int]()
        self.right_children = List[Int]()
        self.feature_count = 0

    def predict_label(read self, read features: List[MetricScalar]) raises -> Int:
        if len(self.predicted_labels) == 0:
            raise Error("adaptive cutoff tree must contain at least one node")
        if len(features) != self.feature_count:
            raise Error("adaptive cutoff tree feature count mismatch")

        var node_index = 0
        while True:
            var split_feature = self.split_feature_indices[node_index]
            if split_feature == -1:
                return self.predicted_labels[node_index]
            if features[split_feature] <= self.split_thresholds[node_index]:
                node_index = self.left_children[node_index]
            else:
                node_index = self.right_children[node_index]


def append_tree_node(
    mut tree: AdaptiveCutoffDecisionTree, predicted_label: Int
) -> Int:
    var node_index = len(tree.predicted_labels)
    tree.split_feature_indices.append(-1)
    tree.split_thresholds.append(zero_metric_scalar())
    tree.predicted_labels.append(predicted_label)
    tree.left_children.append(-1)
    tree.right_children.append(-1)
    return node_index


def all_samples_share_label(
    read labels: List[Int], read sample_indices: List[Int]
) -> Bool:
    if len(sample_indices) <= 1:
        return True
    var first_label = labels[sample_indices[0]]
    for sample_index in range(1, len(sample_indices)):
        if labels[sample_indices[sample_index]] != first_label:
            return False
    return True


def majority_label(
    read labels: List[Int],
    read sample_indices: List[Int],
    label_min: Int,
    label_max: Int,
) raises -> Int:
    if len(sample_indices) == 0:
        raise Error("majority_label requires at least one sample")
    var counts = List[Int]()
    for _ in range(label_max - label_min + 1):
        counts.append(0)
    for sample_index in sample_indices:
        var label = labels[sample_index]
        if label < label_min or label > label_max:
            raise Error("tree label is out of range")
        counts[label - label_min] += 1

    var best_label = label_min
    var best_count = -1
    for offset in range(len(counts)):
        if counts[offset] > best_count:
            best_count = counts[offset]
            best_label = label_min + offset
    return best_label


def gini_impurity(
    read labels: List[Int],
    read sample_indices: List[Int],
    label_min: Int,
    label_max: Int,
) raises -> MetricScalar:
    if len(sample_indices) == 0:
        return zero_metric_scalar()

    var counts = List[Int]()
    for _ in range(label_max - label_min + 1):
        counts.append(0)
    for sample_index in sample_indices:
        counts[labels[sample_index] - label_min] += 1

    var total = MetricScalar(len(sample_indices))
    var impurity = one_metric_scalar()
    for count in counts:
        if count == 0:
            continue
        var probability = MetricScalar(count) / total
        impurity -= probability * probability
    return impurity


def train_tree_node(
    mut tree: AdaptiveCutoffDecisionTree,
    read feature_rows: List[List[MetricScalar]],
    read labels: List[Int],
    read sample_indices: List[Int],
    depth: Int,
    max_depth: Int,
    min_samples_leaf: Int,
    label_min: Int,
    label_max: Int,
) raises -> Int:
    var predicted_label = majority_label(
        labels, sample_indices, label_min, label_max
    )
    var node_index = append_tree_node(tree, predicted_label)

    if depth >= max_depth:
        return node_index
    if len(sample_indices) < 2 * min_samples_leaf:
        return node_index
    if all_samples_share_label(labels, sample_indices):
        return node_index

    var best_feature = -1
    var best_threshold = zero_metric_scalar()
    var best_score = max_metric_scalar()
    var best_left = List[Int]()
    var best_right = List[Int]()

    for feature_index in range(tree.feature_count):
        var thresholds = List[MetricScalar]()
        for sample_index in sample_indices:
            var value = feature_rows[sample_index][feature_index]
            var seen = False
            for existing in thresholds:
                if existing == value:
                    seen = True
                    break
            if not seen:
                thresholds.append(value)

        for threshold in thresholds:
            var left_indices = List[Int]()
            var right_indices = List[Int]()
            for sample_index in sample_indices:
                if feature_rows[sample_index][feature_index] <= threshold:
                    left_indices.append(sample_index)
                else:
                    right_indices.append(sample_index)
            if len(left_indices) < min_samples_leaf:
                continue
            if len(right_indices) < min_samples_leaf:
                continue

            var left_weight = MetricScalar(len(left_indices)) / MetricScalar(
                len(sample_indices)
            )
            var right_weight = MetricScalar(len(right_indices)) / MetricScalar(
                len(sample_indices)
            )
            var split_score = (
                left_weight
                * gini_impurity(labels, left_indices, label_min, label_max)
                + right_weight
                * gini_impurity(labels, right_indices, label_min, label_max)
            )

            if split_score < best_score:
                best_feature = feature_index
                best_threshold = threshold
                best_score = split_score
                best_left = left_indices^
                best_right = right_indices^
    if best_feature == -1:
        return node_index

    tree.split_feature_indices[node_index] = best_feature
    tree.split_thresholds[node_index] = best_threshold
    var left_child = train_tree_node(
        tree,
        feature_rows,
        labels,
        best_left,
        depth + 1,
        max_depth,
        min_samples_leaf,
        label_min,
        label_max,
    )
    var right_child = train_tree_node(
        tree,
        feature_rows,
        labels,
        best_right,
        depth + 1,
        max_depth,
        min_samples_leaf,
        label_min,
        label_max,
    )
    tree.left_children[node_index] = left_child
    tree.right_children[node_index] = right_child
    return node_index


def build_adaptive_cutoff_decision_tree(
    read feature_rows: List[List[MetricScalar]],
    read labels: List[Int],
    max_depth: Int,
    min_samples_leaf: Int = 1,
) raises -> AdaptiveCutoffDecisionTree:
    if len(feature_rows) == 0:
        raise Error("adaptive cutoff tree requires at least one feature row")
    if len(feature_rows) != len(labels):
        raise Error("adaptive cutoff tree features and labels must match")
    if max_depth < 0:
        raise Error("adaptive cutoff tree max_depth must be non-negative")
    if min_samples_leaf <= 0:
        raise Error("adaptive cutoff tree min_samples_leaf must be positive")

    var feature_count = len(feature_rows[0])
    if feature_count == 0:
        raise Error("adaptive cutoff tree requires at least one feature")
    var label_min = labels[0]
    var label_max = labels[0]
    for row in feature_rows:
        if len(row) != feature_count:
            raise Error("adaptive cutoff tree feature rows must be aligned")
    for label in labels:
        if label < label_min:
            label_min = label
        if label > label_max:
            label_max = label

    var tree = AdaptiveCutoffDecisionTree()
    tree.feature_count = feature_count
    var sample_indices = List[Int]()
    for sample_index in range(len(labels)):
        sample_indices.append(sample_index)
    _ = train_tree_node(
        tree,
        feature_rows,
        labels,
        sample_indices,
        0,
        max_depth,
        min_samples_leaf,
        label_min,
        label_max,
    )
    return tree^
