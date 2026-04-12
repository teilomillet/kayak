# Stage-1 candidate generator contract.


struct CandidateGenerator(Copyable):
    var kind: String

    def __init__(out self):
        self.kind = "exact_full_scan"

    def __init__(out self, var kind: String) raises:
        if (
            kind != "exact_full_scan"
            and kind != "document_proxy"
            and kind != "centroid_postings"
            and kind != "centroid_postings_flat"
            and kind != "centroid_heads"
            and kind != "centroid_postings_head"
            and kind != "centroid_postings_head_auto"
            and kind != "centroid_postings_blockmax"
            and kind != "centroid_postings_imputed"
            and kind != "centroid_postings_imputed_flat"
        ):
            raise Error("unknown candidate generator kind: " + kind)

        self.kind = kind^


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
