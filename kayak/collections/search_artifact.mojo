from std.collections import List

from .validation import require_non_empty_string


comptime SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS = "centroid_heads"
comptime SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS = "centroid_postings"
comptime SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX = "document_filter_index"
comptime SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA = "document_metadata"
comptime SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY = "document_proxy"
comptime SEARCH_ARTIFACT_FAMILY_GEM_GRAPH = "gem_graph"
comptime SEARCH_ARTIFACT_FAMILY_LATENT_PROXY = "latent_proxy"


struct SearchArtifactManifest(Copyable):
    var family: String
    var root: String

    def __init__(out self, var family: String, var root: String) raises:
        self.family = require_non_empty_string(family, "search_artifact family")
        self.root = require_non_empty_string(root, "search_artifact root")


def centroid_heads_search_artifact(root: String) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS, root)


def centroid_postings_search_artifact(root: String) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS, root)


def document_proxy_search_artifact(root: String) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY, root)


def latent_proxy_search_artifact(root: String) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(SEARCH_ARTIFACT_FAMILY_LATENT_PROXY, root)


def document_filter_index_search_artifact(
    root: String
) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX, root)


def document_metadata_search_artifact(root: String) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA, root)


def gem_graph_search_artifact(root: String) raises -> SearchArtifactManifest:
    return SearchArtifactManifest(SEARCH_ARTIFACT_FAMILY_GEM_GRAPH, root)


def search_artifact_root(
    read artifacts: List[SearchArtifactManifest], family: String
) -> String:
    for artifact in artifacts:
        if artifact.family == family:
            return artifact.root.copy()

    return ""


def has_search_artifact(
    read artifacts: List[SearchArtifactManifest], family: String
) -> Bool:
    return search_artifact_root(artifacts, family).byte_length() != 0


def require_unique_search_artifact_families(
    read artifacts: List[SearchArtifactManifest]
) raises:
    for left_index in range(len(artifacts)):
        for right_index in range(left_index + 1, len(artifacts)):
            if artifacts[left_index].family == artifacts[right_index].family:
                raise Error(
                    "duplicate search artifact family: "
                    + artifacts[left_index].family
                )


def copy_search_artifacts(
    read artifacts: List[SearchArtifactManifest]
) raises -> List[SearchArtifactManifest]:
    require_unique_search_artifact_families(artifacts)

    var copied = List[SearchArtifactManifest]()
    for artifact in artifacts:
        copied.append(artifact.copy())
    return copied^


def same_search_artifacts(
    read left: List[SearchArtifactManifest], read right: List[SearchArtifactManifest]
) -> Bool:
    if len(left) != len(right):
        return False

    for index in range(len(left)):
        if left[index].family != right[index].family:
            return False
        if left[index].root != right[index].root:
            return False

    return True
