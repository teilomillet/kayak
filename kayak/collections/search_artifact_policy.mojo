from std.collections import List

from kayak.index import (
    DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
)

from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
    SearchArtifactManifest,
)
from .validation import require_non_empty_string


comptime SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET = "centroid_budget"
comptime SEARCH_ARTIFACT_BUILD_CONFIG_CLUSTER_CUTOFF = "cluster_cutoff"
comptime SEARCH_ARTIFACT_BUILD_CONFIG_COARSE_CLUSTER_COUNT = "coarse_cluster_count"
comptime SEARCH_ARTIFACT_BUILD_CONFIG_CONSTRUCTION_NEIGHBOR_COUNT = (
    "construction_neighbor_count"
)
comptime SEARCH_ARTIFACT_BUILD_CONFIG_DEGREE_LIMIT = "degree_limit"
comptime SEARCH_ARTIFACT_BUILD_CONFIG_DOCUMENT_VECTOR_BUDGET = (
    "document_vector_budget"
)
comptime SEARCH_ARTIFACT_BUILD_CONFIG_FINE_CLUSTER_COUNT = "fine_cluster_count"
comptime SEARCH_ARTIFACT_BUILD_CONFIG_POSTING_CAP = "posting_cap"


struct SearchArtifactBuildConfigEntry(Copyable):
    var key: String
    var value: String

    def __init__(out self, var key: String, var value: String) raises:
        self.key = require_non_empty_string(
            key, "search_artifact build config key"
        )
        self.value = require_non_empty_string(
            value, "search_artifact build config value"
        )


def require_unique_search_artifact_build_config_keys(
    read entries: List[SearchArtifactBuildConfigEntry]
) raises:
    for left_index in range(len(entries)):
        for right_index in range(left_index + 1, len(entries)):
            if entries[left_index].key == entries[right_index].key:
                raise Error(
                    "duplicate search artifact build config key: "
                    + entries[left_index].key
                )


def copy_search_artifact_build_config_entries(
    read entries: List[SearchArtifactBuildConfigEntry]
) raises -> List[SearchArtifactBuildConfigEntry]:
    require_unique_search_artifact_build_config_keys(entries)

    var copied = List[SearchArtifactBuildConfigEntry]()
    for entry in entries:
        copied.append(entry.copy())
    return copied^


def same_search_artifact_build_config_entries(
    read left: List[SearchArtifactBuildConfigEntry],
    read right: List[SearchArtifactBuildConfigEntry],
) -> Bool:
    if len(left) != len(right):
        return False

    for index in range(len(left)):
        if left[index].key != right[index].key:
            return False
        if left[index].value != right[index].value:
            return False

    return True


struct SearchArtifactBuildSpec(Copyable):
    var family: String
    var root: String
    var config: List[SearchArtifactBuildConfigEntry]

    def __init__(out self, var family: String, var root: String) raises:
        self = SearchArtifactBuildSpec(family, root, [])

    def __init__(
        out self,
        var family: String,
        var root: String,
        read config: List[SearchArtifactBuildConfigEntry],
    ) raises:
        self.family = require_non_empty_string(
            family, "search_artifact build family"
        )
        self.root = require_non_empty_string(root, "search_artifact build root")
        self.config = copy_search_artifact_build_config_entries(config)


def search_artifact_build_config_value(
    read spec: SearchArtifactBuildSpec, key: String
) -> String:
    for entry in spec.config:
        if entry.key == key:
            return entry.value.copy()

    return ""


def require_unique_search_artifact_build_families(
    read specs: List[SearchArtifactBuildSpec]
) raises:
    for left_index in range(len(specs)):
        for right_index in range(left_index + 1, len(specs)):
            if specs[left_index].family == specs[right_index].family:
                raise Error(
                    "duplicate search artifact build family: "
                    + specs[left_index].family
                )


def require_unique_search_artifact_build_roots(
    read specs: List[SearchArtifactBuildSpec]
) raises:
    for left_index in range(len(specs)):
        for right_index in range(left_index + 1, len(specs)):
            if specs[left_index].root == specs[right_index].root:
                raise Error(
                    "duplicate search artifact build root: "
                    + specs[left_index].root
                )


def copy_search_artifact_build_specs(
    read specs: List[SearchArtifactBuildSpec]
) raises -> List[SearchArtifactBuildSpec]:
    require_unique_search_artifact_build_families(specs)
    require_unique_search_artifact_build_roots(specs)

    var copied = List[SearchArtifactBuildSpec]()
    for spec in specs:
        copied.append(spec.copy())
    return copied^


struct SearchArtifactBuildPolicy(Copyable):
    var stage1_artifacts: List[SearchArtifactBuildSpec]

    def __init__(out self, read stage1_artifacts: List[SearchArtifactBuildSpec]) raises:
        self.stage1_artifacts = copy_search_artifact_build_specs(stage1_artifacts)


def same_search_artifact_build_specs(
    read left: List[SearchArtifactBuildSpec], read right: List[SearchArtifactBuildSpec]
) -> Bool:
    if len(left) != len(right):
        return False

    for index in range(len(left)):
        if left[index].family != right[index].family:
            return False
        if left[index].root != right[index].root:
            return False
        if not same_search_artifact_build_config_entries(
            left[index].config, right[index].config
        ):
            return False

    return True


def same_search_artifact_build_policy(
    read left: SearchArtifactBuildPolicy, read right: SearchArtifactBuildPolicy
) -> Bool:
    return same_search_artifact_build_specs(
        left.stage1_artifacts, right.stage1_artifacts
    )


def search_artifact_build_policy_has_family(
    read policy: SearchArtifactBuildPolicy, family: String
) -> Bool:
    for spec in policy.stage1_artifacts:
        if spec.family == family:
            return True

    return False


def search_artifact_build_policy_supports_required_families(
    read policy: SearchArtifactBuildPolicy,
    read required_families: List[String],
) -> Bool:
    for family in required_families:
        if not search_artifact_build_policy_has_family(policy, family):
            return False

    return True


def document_proxy_build_spec(
    root: String = SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    document_vector_budget: Int = 0,
) raises -> SearchArtifactBuildSpec:
    var config = List[SearchArtifactBuildConfigEntry]()
    if document_vector_budget != 0:
        config.append(
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_DOCUMENT_VECTOR_BUDGET,
                String(document_vector_budget),
            )
        )
    return SearchArtifactBuildSpec(
        SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
        root,
        config,
    )


def centroid_postings_build_spec(
    root: String = SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    centroid_budget: Int = 0,
) raises -> SearchArtifactBuildSpec:
    var config = List[SearchArtifactBuildConfigEntry]()
    if centroid_budget != 0:
        config.append(
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
                String(centroid_budget),
            )
        )
    return SearchArtifactBuildSpec(
        SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
        root,
        config,
    )


def centroid_heads_build_spec(
    posting_cap: Int,
    root: String = SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    centroid_budget: Int = 0,
) raises -> SearchArtifactBuildSpec:
    var config = List[SearchArtifactBuildConfigEntry]()
    if centroid_budget != 0:
        config.append(
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
                String(centroid_budget),
            )
        )
    config.append(
        SearchArtifactBuildConfigEntry(
            SEARCH_ARTIFACT_BUILD_CONFIG_POSTING_CAP,
            String(posting_cap),
        )
    )
    return SearchArtifactBuildSpec(
        SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
        root,
        config,
    )


def gem_graph_build_spec(
    fine_cluster_count: Int,
    coarse_cluster_count: Int,
    cluster_cutoff: Int,
    root: String = SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
    construction_neighbor_count: Int = DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    degree_limit: Int = DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
) raises -> SearchArtifactBuildSpec:
    return SearchArtifactBuildSpec(
        SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
        root,
        [
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_FINE_CLUSTER_COUNT,
                String(fine_cluster_count),
            ),
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_COARSE_CLUSTER_COUNT,
                String(coarse_cluster_count),
            ),
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_CLUSTER_CUTOFF,
                String(cluster_cutoff),
            ),
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_CONSTRUCTION_NEIGHBOR_COUNT,
                String(construction_neighbor_count),
            ),
            SearchArtifactBuildConfigEntry(
                SEARCH_ARTIFACT_BUILD_CONFIG_DEGREE_LIMIT,
                String(degree_limit),
            ),
        ],
    )


def default_search_artifact_build_policy() raises -> SearchArtifactBuildPolicy:
    return SearchArtifactBuildPolicy(
        [
            document_proxy_build_spec(),
            centroid_postings_build_spec(),
        ]
    )


def require_search_artifact_build_policy_layout_safe(
    read policy: SearchArtifactBuildPolicy
) raises:
    for spec in policy.stage1_artifacts:
        if (
            spec.root == "packed_index"
            or spec.root == "text_corpus"
            or spec.root == "document_filter_index"
            or spec.root == "document_metadata"
        ):
            raise Error(
                "search artifact build root conflicts with reserved segment root: "
                + spec.root
            )


def build_spec_as_search_artifact_manifest(
    read spec: SearchArtifactBuildSpec
) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(spec.family.copy(), spec.root.copy())
