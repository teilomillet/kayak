# Stage-1 candidate generator contract.

from kayak.index import (
    DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
    DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
)

from .stage1_capabilities import (
    stage1_capabilities_for_candidate_generator_kind,
)


comptime CANDIDATE_GENERATOR_FAMILY_CENTROID = "centroid"
comptime CANDIDATE_GENERATOR_FAMILY_EXACT = "exact"
comptime CANDIDATE_GENERATOR_FAMILY_GRAPH = "graph"
comptime CANDIDATE_GENERATOR_FAMILY_PROXY = "proxy"


def artifact_family_for_generator_kind(kind: String) raises -> String:
    var capabilities = stage1_capabilities_for_candidate_generator_kind(kind)
    if len(capabilities.required_search_artifact_families) == 1:
        return capabilities.required_search_artifact_families[0].copy()
    return ""


def family_for_generator_kind(kind: String) raises -> String:
    return stage1_capabilities_for_candidate_generator_kind(kind).generator_family.copy()


struct CandidateGenerator(Copyable):
    var kind: String
    var family: String
    var artifact_family: String
    var cluster_top_k_per_query_token: Int
    var beam_width: Int

    def __init__(out self):
        self.kind = "exact_full_scan"
        self.family = CANDIDATE_GENERATOR_FAMILY_EXACT
        self.artifact_family = ""
        self.cluster_top_k_per_query_token = 0
        self.beam_width = 0

    def __init__(
        out self,
        var kind: String,
        cluster_top_k_per_query_token: Int = 0,
        beam_width: Int = 0,
    ) raises:
        if cluster_top_k_per_query_token < 0:
            raise Error(
                "candidate generator cluster_top_k_per_query_token must be non-negative"
            )
        if beam_width < 0:
            raise Error("candidate generator beam_width must be non-negative")
        self.kind = kind^
        self.family = family_for_generator_kind(self.kind)
        self.artifact_family = artifact_family_for_generator_kind(self.kind)
        if self.kind == "gem_graph":
            if cluster_top_k_per_query_token <= 0:
                raise Error(
                    "gem_graph candidate generator requires cluster_top_k_per_query_token > 0"
                )
            if beam_width <= 0:
                raise Error("gem_graph candidate generator requires beam_width > 0")
        elif cluster_top_k_per_query_token != 0 or beam_width != 0:
            raise Error(
                "only gem_graph candidate generators currently accept graph search parameters"
            )
        self.cluster_top_k_per_query_token = cluster_top_k_per_query_token
        self.beam_width = beam_width


def exact_full_scan_candidate_generator() -> CandidateGenerator:
    return CandidateGenerator()


def document_proxy_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("document_proxy")


def centroid_postings_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_postings")


def centroid_postings_flat_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_postings_flat")


def centroid_heads_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_heads")


def centroid_postings_head_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_postings_head")


def centroid_postings_head_auto_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_postings_head_auto")


def centroid_postings_blockmax_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_postings_blockmax")


def centroid_postings_imputed_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_postings_imputed")


def centroid_postings_imputed_flat_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("centroid_postings_imputed_flat")


def gem_graph_candidate_generator(
    cluster_top_k_per_query_token: Int = DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
    beam_width: Int = DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
) raises -> CandidateGenerator:
    return CandidateGenerator(
        "gem_graph", cluster_top_k_per_query_token, beam_width
    )
