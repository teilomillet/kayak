from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.filters import FilterExpression, match_all_filter
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
from .graph_search_counters import GraphSearchCounters
from .score_histogram import build_score_histogram
from .search_plan import SearchPlan
from .stage_profile import SearchStageProfile


struct CollectionSearchExplain(Copyable):
    var collection_id: String
    var snapshot_id: String
    var plan: SearchPlan
    var candidate_set: CandidateSet
    var candidate_stage: SearchStageProfile
    var stage2: SearchStageProfile
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
        stage2: SearchStageProfile,
        candidate_recall_at_final_k: MetricScalar,
        faithfulness: FaithfulnessAssessment,
        var final_hits: List[CollectionHit],
    ):
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.plan = plan.copy()
        self.candidate_set = candidate_set.copy()
        self.candidate_stage = candidate_stage.copy()
        self.stage2 = stage2.copy()
        self.exact_stage = stage2.copy()
        self.candidate_recall_at_final_k = candidate_recall_at_final_k
        self.faithfulness = faithfulness.copy()
        self.final_hits = final_hits^


def explain_collection_search[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
    query_text: String = "",
) raises -> CollectionSearchExplain:
    var candidate_set = candidate_generation_for_plan(
        backend, query, snapshot, plan, filter_expression
    )
    var stage2_result = final_hits_for_plan(
        backend, query, query_text, snapshot, candidate_set, plan
    )
    var oracle_final_hits = exact_oracle_hits_for_snapshot(
        backend,
        query,
        snapshot,
        plan.candidate_budget.final_k,
        filter_expression,
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
            candidate_set.graph_search_counters,
            build_score_histogram(candidate_set.hits, 8),
        ),
        SearchStageProfile(
            plan.stage2_operator.kind.copy(),
            len(candidate_set.hits),
            len(stage2_result.final_hits),
            stage2_result.segment_count,
            stage2_result.document_count,
            stage2_result.token_count,
            stage2_result.vector_count,
            stage2_result.byte_size,
            GraphSearchCounters(),
            build_score_histogram(stage2_result.final_hits, 8),
        ),
        observed_candidate_recall_at_final_k,
        assess_faithfulness(
            plan.faithfulness_policy,
            plan.candidate_generator.kind,
            observed_candidate_recall_at_final_k,
        ),
        stage2_result.final_hits.copy(),
    )
