from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.filters import FilterExpression, match_all_filter
from kayak.numeric import MetricScalar
from kayak.runtime import ExactCpuBackend
from kayak.runtime import ExactScoringBackend

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .execution import (
    candidate_generation_for_plan,
    candidate_recall_at_final_k,
)
from .exact_stage import (
    exact_oracle_hits_for_snapshot,
    exact_oracle_stage_profile_for_snapshot,
)
from .execution_stage2 import stage2_result_for_plan
from .execution_stage3 import stage3_result_for_plan
from .faithfulness import FaithfulnessAssessment, assess_faithfulness
from .score_histogram import build_score_histogram
from .search_plan import SearchPlan
from .serving_scope import (
    SearchServingScope,
    search_serving_scope_for_collection,
)
from .stage_profile import SearchStageProfile


struct CollectionSearchExplain(Copyable):
    var collection_id: String
    var snapshot_id: String
    var serving_scope: SearchServingScope
    var plan: SearchPlan
    var candidate_set: CandidateSet
    var candidate_stage: SearchStageProfile
    var stage2: SearchStageProfile
    var stage3_verifier: SearchStageProfile
    var exact_stage: SearchStageProfile
    var candidate_recall_at_final_k: MetricScalar
    var faithfulness: FaithfulnessAssessment
    var final_hits: List[CollectionHit]

    def __init__(
        out self,
        var collection_id: String,
        var snapshot_id: String,
        serving_scope: SearchServingScope,
        plan: SearchPlan,
        candidate_set: CandidateSet,
        candidate_stage: SearchStageProfile,
        stage2: SearchStageProfile,
        stage3_verifier: SearchStageProfile,
        exact_stage: SearchStageProfile,
        candidate_recall_at_final_k: MetricScalar,
        faithfulness: FaithfulnessAssessment,
        var final_hits: List[CollectionHit],
    ):
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.serving_scope = serving_scope.copy()
        self.plan = plan.copy()
        self.candidate_set = candidate_set.copy()
        self.candidate_stage = candidate_stage.copy()
        self.stage2 = stage2.copy()
        self.stage3_verifier = stage3_verifier.copy()
        self.exact_stage = exact_stage.copy()
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
    var stage2_result = stage2_result_for_plan(
        backend, query, query_text, snapshot, candidate_set, plan
    )
    var stage3_result = stage3_result_for_plan(
        query_text,
        snapshot,
        stage2_result,
        plan,
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
    var candidate_stage_score_histogram = build_score_histogram(candidate_set.hits, 8)
    var stage2_score_histogram = build_score_histogram(stage2_result.final_hits, 8)
    var stage3_score_histogram = build_score_histogram(stage3_result.final_hits, 8)
    var candidate_stage = SearchStageProfile(
        "candidate_generation",
        snapshot.snapshot.stats.document_count,
        len(candidate_set.hits),
        candidate_set.segment_count,
        candidate_set.document_count,
        candidate_set.token_count,
        candidate_set.vector_count,
        candidate_set.byte_size,
        candidate_stage_score_histogram,
    )
    if candidate_set.tracks_graph_search:
        candidate_stage = SearchStageProfile(
            "candidate_generation",
            snapshot.snapshot.stats.document_count,
            len(candidate_set.hits),
            candidate_set.segment_count,
            candidate_set.document_count,
            candidate_set.token_count,
            candidate_set.vector_count,
            candidate_set.byte_size,
            candidate_set.graph_search_counters,
            candidate_stage_score_histogram,
        )
    var stage2 = SearchStageProfile(
        plan.stage2_reference_operator.kind.copy(),
        len(candidate_set.hits),
        len(stage2_result.final_hits),
        stage2_result.segment_count,
        stage2_result.document_count,
        stage2_result.token_count,
        stage2_result.vector_count,
        stage2_result.byte_size,
        stage2_score_histogram,
        stage2_result.materialized_artifacts.copy(),
    )
    var stage3 = SearchStageProfile(
        plan.stage3_verifier.kind.copy(),
        len(stage2_result.final_hits),
        len(stage3_result.final_hits),
        stage3_result.segment_count,
        stage3_result.document_count,
        stage3_result.token_count,
        stage3_result.vector_count,
        stage3_result.byte_size,
        stage3_score_histogram,
        stage3_result.materialized_artifacts.copy(),
    )

    return CollectionSearchExplain(
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        search_serving_scope_for_collection(snapshot.collection),
        plan,
        candidate_set.copy(),
        candidate_stage,
        stage2,
        stage3,
        exact_oracle_stage_profile_for_snapshot(snapshot, oracle_final_hits),
        observed_candidate_recall_at_final_k,
        assess_faithfulness(
            plan.faithfulness_policy,
            plan.candidate_generator.kind,
            observed_candidate_recall_at_final_k,
        ),
        stage3_result.final_hits.copy(),
    )


def explain_collection_search(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
    query_text: String = "",
) raises -> CollectionSearchExplain:
    var candidate_set = candidate_generation_for_plan(
        backend, query, snapshot, plan, filter_expression
    )
    var stage2_result = stage2_result_for_plan(
        backend, query, query_text, snapshot, candidate_set, plan
    )
    var stage3_result = stage3_result_for_plan(
        query_text,
        snapshot,
        stage2_result,
        plan,
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
    var candidate_stage_score_histogram = build_score_histogram(candidate_set.hits, 8)
    var stage2_score_histogram = build_score_histogram(stage2_result.final_hits, 8)
    var stage3_score_histogram = build_score_histogram(stage3_result.final_hits, 8)
    var candidate_stage = SearchStageProfile(
        "candidate_generation",
        snapshot.snapshot.stats.document_count,
        len(candidate_set.hits),
        candidate_set.segment_count,
        candidate_set.document_count,
        candidate_set.token_count,
        candidate_set.vector_count,
        candidate_set.byte_size,
        candidate_stage_score_histogram,
    )
    if candidate_set.tracks_graph_search:
        candidate_stage = SearchStageProfile(
            "candidate_generation",
            snapshot.snapshot.stats.document_count,
            len(candidate_set.hits),
            candidate_set.segment_count,
            candidate_set.document_count,
            candidate_set.token_count,
            candidate_set.vector_count,
            candidate_set.byte_size,
            candidate_set.graph_search_counters,
            candidate_stage_score_histogram,
        )
    var stage2 = SearchStageProfile(
        plan.stage2_reference_operator.kind.copy(),
        len(candidate_set.hits),
        len(stage2_result.final_hits),
        stage2_result.segment_count,
        stage2_result.document_count,
        stage2_result.token_count,
        stage2_result.vector_count,
        stage2_result.byte_size,
        stage2_score_histogram,
        stage2_result.materialized_artifacts.copy(),
    )
    var stage3 = SearchStageProfile(
        plan.stage3_verifier.kind.copy(),
        len(stage2_result.final_hits),
        len(stage3_result.final_hits),
        stage3_result.segment_count,
        stage3_result.document_count,
        stage3_result.token_count,
        stage3_result.vector_count,
        stage3_result.byte_size,
        stage3_score_histogram,
        stage3_result.materialized_artifacts.copy(),
    )

    return CollectionSearchExplain(
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        search_serving_scope_for_collection(snapshot.collection),
        plan,
        candidate_set.copy(),
        candidate_stage,
        stage2,
        stage3,
        exact_oracle_stage_profile_for_snapshot(snapshot, oracle_final_hits),
        observed_candidate_recall_at_final_k,
        assess_faithfulness(
            plan.faithfulness_policy,
            plan.candidate_generator.kind,
            observed_candidate_recall_at_final_k,
        ),
        stage3_result.final_hits.copy(),
    )
