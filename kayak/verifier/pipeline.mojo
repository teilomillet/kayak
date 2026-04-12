from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.runtime import ExactScoringBackend
from kayak.search import SearchHit, search_exact

from .exact_late_interaction import rerank_hits_exact_late_interaction
from .noop_verifier import rerank_hits_noop
from .verifier_reranker import (
    VerifierReranker,
    effective_candidate_k,
)


def rerank_hits_with_verifier(
    read query: EncodedQuery,
    read index: PackedIndex,
    read hits: List[SearchHit],
    k: Int,
    read verifier: VerifierReranker,
) raises -> List[SearchHit]:
    if verifier.kind == "none":
        return rerank_hits_noop(hits, k)

    if verifier.kind == "exact_late_interaction":
        return rerank_hits_exact_late_interaction(query, index, hits, k)

    raise Error("unknown verifier kind: " + verifier.kind)


def search_exact_with_verifier[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read index: PackedIndex,
    k: Int,
    read verifier: VerifierReranker,
) raises -> List[SearchHit]:
    var candidate_hits = search_exact(
        backend, query, index, effective_candidate_k(verifier, k)
    )
    return rerank_hits_with_verifier(query, index, candidate_hits, k, verifier)
