from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.runtime import ExactScoringBackend

from .candidate_set import CandidateSet
from .search_plan import SearchPlan


def candidate_generation_for_graph_family[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> CandidateSet:
    _ = backend
    _ = query
    _ = snapshot

    raise Error(
        "graph stage-1 family is registered but not implemented: "
        + plan.candidate_generator.kind
    )
