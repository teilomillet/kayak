# Stage-1 candidate generator contract.

from std.collections import List

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
    return stage1_capabilities_for_candidate_generator_kind(
        kind
    ).single_required_search_artifact_family()


def family_for_generator_kind(kind: String) raises -> String:
    return stage1_capabilities_for_candidate_generator_kind(kind).generator_family.copy()


struct CandidateGenerator(Copyable):
    var kind: String
    var family: String
    var artifact_family: String
    var interaction_semantics: String
    var alignment_granularity: String
    var score_kind: String
    var required_search_artifact_families: List[String]
    var is_exact: Bool
    var supports_match_all_filter: Bool
    var supports_structured_filter: Bool
    var cluster_top_k_per_query_token: Int
    var beam_width: Int

    def __init__(out self):
        self.kind = "exact_full_scan"
        self.family = CANDIDATE_GENERATOR_FAMILY_EXACT
        self.artifact_family = ""
        self.interaction_semantics = "exact_late_interaction"
        self.alignment_granularity = "document_tokens"
        self.score_kind = "exact_score"
        self.required_search_artifact_families = List[String]()
        self.is_exact = True
        self.supports_match_all_filter = True
        self.supports_structured_filter = True
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
        var capabilities = stage1_capabilities_for_candidate_generator_kind(kind)
        self.kind = kind^
        self.family = capabilities.generator_family.copy()
        self.artifact_family = capabilities.single_required_search_artifact_family()
        self.interaction_semantics = capabilities.interaction_semantics.copy()
        self.alignment_granularity = capabilities.alignment_granularity.copy()
        self.score_kind = capabilities.score_kind.copy()
        self.required_search_artifact_families = (
            capabilities.required_search_artifact_families.copy()
        )
        self.is_exact = capabilities.stage1_is_exact
        self.supports_match_all_filter = capabilities.supports_match_all_filter
        self.supports_structured_filter = capabilities.supports_structured_filter
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

    def requires_artifact_family(self, family: String) -> Bool:
        for required_family in self.required_search_artifact_families:
            if required_family == family:
                return True

        return False


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
