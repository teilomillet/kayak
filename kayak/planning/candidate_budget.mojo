# Budget contracts for multi-stage search.


struct CandidateBudget(Copyable):
    var final_k: Int
    var candidate_k: Int

    def __init__(out self, final_k: Int, candidate_k: Int) raises:
        if final_k < 0:
            raise Error("final_k must be non-negative")

        if candidate_k < 0:
            raise Error("candidate_k must be non-negative")

        if candidate_k < final_k:
            raise Error("candidate_k must be greater than or equal to final_k")

        self.final_k = final_k
        self.candidate_k = candidate_k
