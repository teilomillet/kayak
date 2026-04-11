struct VerifierReranker(Copyable):
    var kind: String
    var candidate_k: Int

    def __init__(out self):
        self.kind = "none"
        self.candidate_k = 0

    def __init__(out self, var kind: String, candidate_k: Int) raises:
        if kind != "none" and kind != "exact_late_interaction":
            raise Error("unknown verifier kind: " + kind)

        if candidate_k < 0:
            raise Error("verifier candidate_k must be non-negative")

        self.kind = kind^
        self.candidate_k = candidate_k


def no_verifier() -> VerifierReranker:
    return VerifierReranker()


def exact_late_interaction_verifier(
    candidate_k: Int
) raises -> VerifierReranker:
    return VerifierReranker("exact_late_interaction", candidate_k)


def effective_candidate_k(
    read verifier: VerifierReranker, final_k: Int
) -> Int:
    if verifier.kind == "none":
        return final_k

    if verifier.candidate_k < final_k:
        return final_k

    return verifier.candidate_k
