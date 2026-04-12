from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.numeric import MetricScalar
from kayak.runtime import ExactScoringBackend

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .execution import (
    candidate_generation_for_plan,
    candidate_recall_at_final_k,
    final_hits_for_plan,
)
from .exact_stage import exact_oracle_hits_for_snapshot
from .faithfulness import FaithfulnessAssessment, assess_faithfulness
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
    var faithfulness: FaithfulnessAssessment
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
        faithfulness: FaithfulnessAssessment,
        var final_hits: List[CollectionHit],
    ):
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.plan = plan.copy()
        self.candidate_set = candidate_set.copy()
        self.candidate_stage = candidate_stage.copy()
        self.exact_stage = exact_stage.copy()
        self.candidate_recall_at_final_k = candidate_recall_at_final_k
        self.faithfulness = faithfulness.copy()
        self.final_hits = final_hits^


def explain_collection_search[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> CollectionSearchExplain:
    var candidate_set = candidate_generation_for_plan(
        backend, query, snapshot, plan
    )
    var exact_stage = final_hits_for_plan(
        backend, query, snapshot, candidate_set, plan
    )
    var oracle_final_hits = exact_oracle_hits_for_snapshot(
        backend, query, snapshot, plan.candidate_budget.final_k
    )
    var observed_candidate_recall_at_final_k = candidate_recall_at_final_k(
        candidate_set, oracle_final_hits
    )

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
            len(exact_stage.final_hits),
            exact_stage.segment_count,
            exact_stage.document_count,
            exact_stage.token_count,
            exact_stage.vector_count,
            exact_stage.byte_size,
            build_score_histogram(exact_stage.final_hits, 8),
        ),
        observed_candidate_recall_at_final_k,
        assess_faithfulness(
            plan.faithfulness_policy,
            plan.candidate_generator.kind,
            observed_candidate_recall_at_final_k,
        ),
        exact_stage.final_hits.copy(),
    )
