from std.collections import List
from std.pathlib import Path

from kayak.index import (
    DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
)
from kayak.storage import (
    StoredPackedIndex,
    build_stored_gem_graph_index,
    centroid_heads_storage_byte_size,
    centroid_postings_storage_byte_size,
    document_proxy_storage_byte_size,
    ensure_stored_centroid_heads_index,
    ensure_stored_centroid_posting_index,
    ensure_stored_document_proxy_index,
    gem_graph_storage_byte_size,
    save_stored_gem_graph_index,
)
from kayak.storage.text_codec import parse_int

from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
)
from .search_artifact_policy import (
    SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
    SEARCH_ARTIFACT_BUILD_CONFIG_CLUSTER_CUTOFF,
    SEARCH_ARTIFACT_BUILD_CONFIG_COARSE_CLUSTER_COUNT,
    SEARCH_ARTIFACT_BUILD_CONFIG_CONSTRUCTION_NEIGHBOR_COUNT,
    SEARCH_ARTIFACT_BUILD_CONFIG_DEGREE_LIMIT,
    SEARCH_ARTIFACT_BUILD_CONFIG_DOCUMENT_VECTOR_BUDGET,
    SEARCH_ARTIFACT_BUILD_CONFIG_FINE_CLUSTER_COUNT,
    SEARCH_ARTIFACT_BUILD_CONFIG_POSTING_CAP,
    SearchArtifactBuildPolicy,
    SearchArtifactBuildSpec,
    search_artifact_build_config_value,
)


def require_allowed_build_config_keys(
    read spec: SearchArtifactBuildSpec, read allowed_keys: List[String]
) raises:
    for entry in spec.config:
        var allowed = False
        for allowed_key in allowed_keys:
            if entry.key == allowed_key:
                allowed = True
                break

        if not allowed:
            raise Error(
                "unsupported search artifact build config key for "
                + spec.family
                + ": "
                + entry.key
            )


def optional_non_negative_build_int(
    read spec: SearchArtifactBuildSpec,
    key: String,
    default_value: Int,
) raises -> Int:
    var value = search_artifact_build_config_value(spec, key)
    if value.byte_length() == 0:
        return default_value

    var parsed = parse_int(value, key)
    if parsed < 0:
        raise Error(
            "search artifact build config "
            + key
            + " must be non-negative for "
            + spec.family
        )
    return parsed


def required_positive_build_int(
    read spec: SearchArtifactBuildSpec, key: String
) raises -> Int:
    var value = search_artifact_build_config_value(spec, key)
    if value.byte_length() == 0:
        raise Error(
            "search artifact build config "
            + key
            + " is required for "
            + spec.family
        )

    var parsed = parse_int(value, key)
    if parsed <= 0:
        raise Error(
            "search artifact build config "
            + key
            + " must be positive for "
            + spec.family
        )
    return parsed


def require_search_artifact_build_spec_supported_for_segment_sealing(
    read spec: SearchArtifactBuildSpec
) raises:
    if spec.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY:
        require_allowed_build_config_keys(
            spec,
            [SEARCH_ARTIFACT_BUILD_CONFIG_DOCUMENT_VECTOR_BUDGET],
        )
        _ = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_DOCUMENT_VECTOR_BUDGET,
            0,
        )
        return

    if spec.family == SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS:
        require_allowed_build_config_keys(
            spec,
            [SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET],
        )
        _ = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
            0,
        )
        return

    if spec.family == SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS:
        require_allowed_build_config_keys(
            spec,
            [
                SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
                SEARCH_ARTIFACT_BUILD_CONFIG_POSTING_CAP,
            ],
        )
        _ = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
            0,
        )
        _ = required_positive_build_int(
            spec, SEARCH_ARTIFACT_BUILD_CONFIG_POSTING_CAP
        )
        return

    if spec.family == SEARCH_ARTIFACT_FAMILY_GEM_GRAPH:
        require_allowed_build_config_keys(
            spec,
            [
                SEARCH_ARTIFACT_BUILD_CONFIG_FINE_CLUSTER_COUNT,
                SEARCH_ARTIFACT_BUILD_CONFIG_COARSE_CLUSTER_COUNT,
                SEARCH_ARTIFACT_BUILD_CONFIG_CLUSTER_CUTOFF,
                SEARCH_ARTIFACT_BUILD_CONFIG_CONSTRUCTION_NEIGHBOR_COUNT,
                SEARCH_ARTIFACT_BUILD_CONFIG_DEGREE_LIMIT,
            ],
        )
        _ = required_positive_build_int(
            spec, SEARCH_ARTIFACT_BUILD_CONFIG_FINE_CLUSTER_COUNT
        )
        _ = required_positive_build_int(
            spec, SEARCH_ARTIFACT_BUILD_CONFIG_COARSE_CLUSTER_COUNT
        )
        _ = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_CLUSTER_CUTOFF,
            0,
        )
        _ = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_CONSTRUCTION_NEIGHBOR_COUNT,
            DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
        )
        if (
            optional_non_negative_build_int(
                spec,
                SEARCH_ARTIFACT_BUILD_CONFIG_CONSTRUCTION_NEIGHBOR_COUNT,
                DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
            )
            <= 0
        ):
            raise Error(
                "search artifact build config construction_neighbor_count must be positive for "
                + spec.family
            )
        if (
            optional_non_negative_build_int(
                spec,
                SEARCH_ARTIFACT_BUILD_CONFIG_DEGREE_LIMIT,
                DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
            )
            <= 0
        ):
            raise Error(
                "search artifact build config degree_limit must be positive for "
                + spec.family
            )
        return

    raise Error(
        "segment sealing does not yet support configured build family: "
        + spec.family
    )


def require_search_artifact_build_policy_supported_for_segment_sealing(
    read policy: SearchArtifactBuildPolicy
) raises:
    for spec in policy.stage1_artifacts:
        require_search_artifact_build_spec_supported_for_segment_sealing(spec)


def build_search_artifact_for_segment(
    segment_root: Path,
    read stored_index: StoredPackedIndex,
    read spec: SearchArtifactBuildSpec,
) raises -> Int:
    require_search_artifact_build_spec_supported_for_segment_sealing(spec)

    if spec.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY:
        var document_vector_budget = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_DOCUMENT_VECTOR_BUDGET,
            0,
        )
        _ = ensure_stored_document_proxy_index(
            segment_root / spec.root,
            stored_index,
            document_vector_budget,
        )
        return document_proxy_storage_byte_size(segment_root / spec.root)

    if spec.family == SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS:
        var centroid_budget = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
            0,
        )
        _ = ensure_stored_centroid_posting_index(
            segment_root / spec.root,
            stored_index,
            centroid_budget,
        )
        return centroid_postings_storage_byte_size(segment_root / spec.root)

    if spec.family == SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS:
        var centroid_budget = optional_non_negative_build_int(
            spec,
            SEARCH_ARTIFACT_BUILD_CONFIG_CENTROID_BUDGET,
            0,
        )
        var posting_cap = required_positive_build_int(
            spec, SEARCH_ARTIFACT_BUILD_CONFIG_POSTING_CAP
        )
        _ = ensure_stored_centroid_heads_index(
            segment_root / spec.root,
            stored_index,
            centroid_budget,
            posting_cap,
        )
        return centroid_heads_storage_byte_size(segment_root / spec.root)

    var fine_cluster_count = required_positive_build_int(
        spec, SEARCH_ARTIFACT_BUILD_CONFIG_FINE_CLUSTER_COUNT
    )
    var coarse_cluster_count = required_positive_build_int(
        spec, SEARCH_ARTIFACT_BUILD_CONFIG_COARSE_CLUSTER_COUNT
    )
    var cluster_cutoff = optional_non_negative_build_int(
        spec,
        SEARCH_ARTIFACT_BUILD_CONFIG_CLUSTER_CUTOFF,
        0,
    )
    var construction_neighbor_count = optional_non_negative_build_int(
        spec,
        SEARCH_ARTIFACT_BUILD_CONFIG_CONSTRUCTION_NEIGHBOR_COUNT,
        DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    )
    if construction_neighbor_count <= 0:
        raise Error(
            "search artifact build config construction_neighbor_count must be positive for "
            + spec.family
        )
    var degree_limit = optional_non_negative_build_int(
        spec,
        SEARCH_ARTIFACT_BUILD_CONFIG_DEGREE_LIMIT,
        DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
    )
    if degree_limit <= 0:
        raise Error(
            "search artifact build config degree_limit must be positive for "
            + spec.family
        )

    save_stored_gem_graph_index(
        segment_root / spec.root,
        build_stored_gem_graph_index(
            stored_index,
            fine_cluster_count,
            coarse_cluster_count,
            cluster_cutoff,
            construction_neighbor_count,
            degree_limit,
        ),
    )
    return gem_graph_storage_byte_size(segment_root / spec.root)
