# Stage-1 candidate generator contract.


struct CandidateGenerator(Copyable):
    var kind: String

    def __init__(out self):
        self.kind = "exact_full_scan"

    def __init__(out self, var kind: String) raises:
        if kind != "exact_full_scan":
            raise Error("unknown candidate generator kind: " + kind)

        self.kind = kind^


def exact_full_scan_candidate_generator() -> CandidateGenerator:
    return CandidateGenerator()
