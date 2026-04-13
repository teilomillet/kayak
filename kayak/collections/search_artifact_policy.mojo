from std.collections import List

from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SearchArtifactManifest,
)
from .validation import require_non_empty_string


struct SearchArtifactBuildSpec(Copyable):
    var family: String
    var root: String

    def __init__(out self, var family: String, var root: String) raises:
        self.family = require_non_empty_string(
            family, "search_artifact build family"
        )
        self.root = require_non_empty_string(root, "search_artifact build root")


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

    return True


def same_search_artifact_build_policy(
    read left: SearchArtifactBuildPolicy, read right: SearchArtifactBuildPolicy
) -> Bool:
    return same_search_artifact_build_specs(
        left.stage1_artifacts, right.stage1_artifacts
    )


def document_proxy_build_spec(
    root: String = SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
) raises -> SearchArtifactBuildSpec:
    return SearchArtifactBuildSpec(SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY, root)


def centroid_postings_build_spec(
    root: String = SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
) raises -> SearchArtifactBuildSpec:
    return SearchArtifactBuildSpec(SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS, root)


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
            or spec.root == "document_metadata"
        ):
            raise Error(
                "search artifact build root conflicts with reserved segment root: "
                + spec.root
            )


def require_search_artifact_build_policy_supported_for_segment_sealing(
    read policy: SearchArtifactBuildPolicy
) raises:
    for spec in policy.stage1_artifacts:
        if (
            spec.family != SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
            and spec.family != SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
        ):
            raise Error(
                "segment sealing does not yet support configured build family: "
                + spec.family
            )


def build_spec_as_search_artifact_manifest(
    read spec: SearchArtifactBuildSpec
) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(spec.family.copy(), spec.root.copy())
