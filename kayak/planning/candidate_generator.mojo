# Stage-1 candidate generator contract.

comptime CANDIDATE_GENERATOR_FAMILY_CENTROID = "centroid"
comptime CANDIDATE_GENERATOR_FAMILY_EXACT = "exact"
comptime CANDIDATE_GENERATOR_FAMILY_GRAPH = "graph"
comptime CANDIDATE_GENERATOR_FAMILY_PROXY = "proxy"


def artifact_family_for_generator_kind(kind: String) raises -> String:
    if kind == "exact_full_scan":
        return ""
    if kind == "document_proxy":
        return "document_proxy"
    if kind == "centroid_heads":
        return "centroid_heads"
    if (
        kind == "centroid_postings"
        or kind == "centroid_postings_flat"
        or kind == "centroid_postings_head"
        or kind == "centroid_postings_head_auto"
        or kind == "centroid_postings_blockmax"
        or kind == "centroid_postings_imputed"
        or kind == "centroid_postings_imputed_flat"
    ):
        return "centroid_postings"
    if kind == "gem_graph":
        return "gem_graph"

    raise Error("unknown candidate generator kind: " + kind)


def family_for_generator_kind(kind: String) raises -> String:
    if kind == "exact_full_scan":
        return CANDIDATE_GENERATOR_FAMILY_EXACT
    if kind == "document_proxy":
        return CANDIDATE_GENERATOR_FAMILY_PROXY
    if (
        kind == "centroid_postings"
        or kind == "centroid_postings_flat"
        or kind == "centroid_heads"
        or kind == "centroid_postings_head"
        or kind == "centroid_postings_head_auto"
        or kind == "centroid_postings_blockmax"
        or kind == "centroid_postings_imputed"
        or kind == "centroid_postings_imputed_flat"
    ):
        return CANDIDATE_GENERATOR_FAMILY_CENTROID
    if kind == "gem_graph":
        return CANDIDATE_GENERATOR_FAMILY_GRAPH

    raise Error("unknown candidate generator kind: " + kind)


struct CandidateGenerator(Copyable):
    var kind: String
    var family: String
    var artifact_family: String

    def __init__(out self):
        self.kind = "exact_full_scan"
        self.family = CANDIDATE_GENERATOR_FAMILY_EXACT
        self.artifact_family = ""

    def __init__(out self, var kind: String) raises:
        self.kind = kind^
        self.family = family_for_generator_kind(self.kind)
        self.artifact_family = artifact_family_for_generator_kind(self.kind)


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


def gem_graph_candidate_generator() raises -> CandidateGenerator:
    return CandidateGenerator("gem_graph")
