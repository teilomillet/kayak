from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.numeric import MetricScalar
from kayak.runtime import ExactCpuBackend

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .execution import (
    candidate_generation_for_plan,
    candidate_recall_at_final_k,
    final_hits_for_plan,
)
from .score_histogram import build_score_histogram
from .search_plan import SearchPlan
from .stage_profile import SearchStageProfile


struct CollectionSearchExplain(Copyable):
    var collection_id: String
    var snapshot_id: String
    var plan: SearchPlan
    var candidate_set: CandidateSet
    var candidate_stage: SearchStageProfile
    var exact_stage: SearchStageProfile
    var candidate_recall_at_final_k: MetricScalar
    var final_hits: List[CollectionHit]

    def __init__(
        out self,
        var collection_id: String,
        var snapshot_id: String,
        plan: SearchPlan,
        candidate_set: CandidateSet,
        candidate_stage: SearchStageProfile,
        exact_stage: SearchStageProfile,
        candidate_recall_at_final_k: MetricScalar,
        var final_hits: List[CollectionHit],
    ):
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.plan = plan.copy()
        self.candidate_set = candidate_set.copy()
        self.candidate_stage = candidate_stage.copy()
        self.exact_stage = exact_stage.copy()
        self.candidate_recall_at_final_k = candidate_recall_at_final_k
        self.final_hits = final_hits^


def explain_collection_search(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> CollectionSearchExplain:
    var candidate_set = candidate_generation_for_plan(
        backend, query, snapshot, plan
    )
    var final_hits = final_hits_for_plan(candidate_set, plan)

    return CollectionSearchExplain(
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        plan,
        candidate_set.copy(),
        SearchStageProfile(
            "candidate_generation",
            snapshot.snapshot.stats.document_count,
            len(candidate_set.hits),
            candidate_set.segment_count,
            candidate_set.document_count,
            candidate_set.token_count,
            candidate_set.vector_count,
            candidate_set.byte_size,
            build_score_histogram(candidate_set.hits, 8),
        ),
        SearchStageProfile(
            "exact_late_interaction",
            len(candidate_set.hits),
            len(final_hits),
            candidate_set.segment_count,
            candidate_set.document_count,
            candidate_set.token_count,
            candidate_set.vector_count,
            candidate_set.byte_size,
            build_score_histogram(final_hits, 8),
        ),
        candidate_recall_at_final_k(candidate_set, final_hits),
        final_hits^,
    )
