# Frontier-policy probe for held-out GEM variants on the first real synthetic
# hard-recall profile.
#
# Owns:
# - a fixed-profile comparison between the current per-entry local frontier and
#   benchmark-only global or hybrid frontier variants over the same cached GEM
#   artifacts
# - exact stage-2 rerank and oracle candidate-recall comparison for all tested
#   frontier policies
#
# Does not own:
# - changes to the serving path
# - graph construction or artifact materialization policy beyond reusing the
#   existing held-out query-time probe caches

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.collections import (
    ResolvedCollectionSnapshot,
    SnapshotId,
    load_resolved_collection_snapshot,
    loaded_segment_stored_gem_graph_index,
)
from kayak.eval import evaluate_query_hits
from kayak.planning import final_hits_to_search_hits
from kayak.planning.collection_hit import CollectionHit
from kayak.planning.graph_frontier_policy import (
    GRAPH_FRONTIER_POLICY_KIND_GLOBAL_BEST_FIRST,
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_FAIR_ROUND,
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_PER_ENTRY,
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_QUOTA2_ROUND,
    GRAPH_FRONTIER_POLICY_KIND_LOCAL_PER_ENTRY,
    graph_frontier_policy_kinds,
)
from kayak.planning.graph_frontier_runtime import (
    segment_hits_for_gem_graph,
)
from kayak.planning.exact_stage import (
    exact_oracle_hits_for_snapshot,
    exact_rerank_candidates_for_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig
from kayak.storage import StoredGemGraphIndex

from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_ablation_json import (
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
    GEM_HELDOUT_VARIANT_BASELINE,
    GemHeldoutQuerySplit,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    gem_heldout_ablation_variant_spec,
)
from .gem_heldout_querytime_probe import (
    gem_heldout_querytime_probe_collection_id,
    gem_heldout_querytime_probe_collection_root,
    materialized_gem_heldout_querytime_probe_collection_root,
)
from .json_common import json_escape
from .synthetic_hard_recall_fixture import (
    default_synthetic_hard_recall_profiles,
    make_synthetic_hard_recall_fixture,
)


comptime GEM_HELDOUT_FRONTIER_POLICY_LOCAL_PER_ENTRY = (
    GRAPH_FRONTIER_POLICY_KIND_LOCAL_PER_ENTRY
)
comptime GEM_HELDOUT_FRONTIER_POLICY_GLOBAL_BEST_FIRST = (
    GRAPH_FRONTIER_POLICY_KIND_GLOBAL_BEST_FIRST
)
comptime GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_PER_ENTRY = (
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_PER_ENTRY
)
comptime GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_FAIR_ROUND = (
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_FAIR_ROUND
)
comptime GEM_HELDOUT_FRONTIER_POLICY_HYBRID_BEST_HEAD_QUOTA2_ROUND = (
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_QUOTA2_ROUND
)


struct GemHeldoutFrontierPolicyProbeSummary(Copyable):
    var slice_name: String
    var variant_kind: String
    var frontier_policy: String
    var candidate_k: Int
    var search_cluster_top_k_per_query_token: Int
    var search_beam_width: Int
    var mean_candidate_recall_at_final_k: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var mean_stage1_graph_visited_vertex_count: Float64
    var mean_stage1_graph_expanded_edge_count: Float64
    var mean_stage1_graph_visited_cluster_count: Float64
    var mean_stage1_graph_entry_point_count: Float64
    var mean_stage1_graph_max_frontier_size: Float64

    def __init__(
        out self,
        var slice_name: String,
        var variant_kind: String,
        var frontier_policy: String,
        candidate_k: Int,
        search_cluster_top_k_per_query_token: Int,
        search_beam_width: Int,
        mean_candidate_recall_at_final_k: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        mean_stage1_graph_visited_vertex_count: Float64,
        mean_stage1_graph_expanded_edge_count: Float64,
        mean_stage1_graph_visited_cluster_count: Float64,
        mean_stage1_graph_entry_point_count: Float64,
        mean_stage1_graph_max_frontier_size: Float64,
    ):
        self.slice_name = slice_name^
        self.variant_kind = variant_kind^
        self.frontier_policy = frontier_policy^
        self.candidate_k = candidate_k
        self.search_cluster_top_k_per_query_token = (
            search_cluster_top_k_per_query_token
        )
        self.search_beam_width = search_beam_width
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.mean_stage1_graph_visited_vertex_count = (
            mean_stage1_graph_visited_vertex_count
        )
        self.mean_stage1_graph_expanded_edge_count = (
            mean_stage1_graph_expanded_edge_count
        )
        self.mean_stage1_graph_visited_cluster_count = (
            mean_stage1_graph_visited_cluster_count
        )
        self.mean_stage1_graph_entry_point_count = (
            mean_stage1_graph_entry_point_count
        )
        self.mean_stage1_graph_max_frontier_size = (
            mean_stage1_graph_max_frontier_size
        )


def gem_heldout_frontier_policy_probe_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def gem_heldout_frontier_policy_probe_candidate_ks() -> List[Int]:
    return [8, 16, 32, 64]


def gem_heldout_frontier_policy_probe_frontier_policies() -> List[String]:
    return graph_frontier_policy_kinds()


def gem_heldout_frontier_policy_probe_search_cluster_top_k() -> Int:
    return 2


def gem_heldout_frontier_policy_probe_beam_width() -> Int:
    return 32


def collection_hit_matches(
    read left: CollectionHit, read right: CollectionHit
) -> Bool:
    return left.segment_id == right.segment_id and left.doc_id == right.doc_id


def collection_hit_recall_at_final_k(
    read candidate_hits: List[CollectionHit], read oracle_final_hits: List[CollectionHit]
) -> Float64:
    if len(oracle_final_hits) == 0:
        return 0.0

    var found_count = 0
    for oracle_hit in oracle_final_hits:
        for candidate_hit in candidate_hits:
            if collection_hit_matches(oracle_hit, candidate_hit):
                found_count += 1
                break
    return Float64(found_count) / Float64(len(oracle_final_hits))


def build_gem_heldout_frontier_policy_probe_summary(
    read backend: ExactCpuBackend,
    read split: GemHeldoutQuerySplit,
    read snapshot: ResolvedCollectionSnapshot,
    read stored_gem: StoredGemGraphIndex,
    variant_kind: String,
    frontier_policy: String,
    candidate_k: Int,
    search_cluster_top_k_per_query_token: Int,
    search_beam_width: Int,
) raises -> GemHeldoutFrontierPolicyProbeSummary:
    var heldout_task = split.evaluation_task.task.copy()
    var segment_id = snapshot.segments[0].manifest.segment_id.value.copy()
    var candidate_recall_total = 0.0
    var recall_total = 0.0
    var success_total = 0.0
    var visited_vertex_total = 0.0
    var expanded_edge_total = 0.0
    var visited_cluster_total = 0.0
    var entry_point_total = 0.0
    var max_frontier_total = 0.0

    for judged_query in heldout_task.queries:
        var stage1 = segment_hits_for_gem_graph(
            judged_query.query,
            segment_id,
            0,
            stored_gem,
            candidate_k,
            search_cluster_top_k_per_query_token,
            search_beam_width,
            frontier_policy,
        )
        var stage2 = exact_rerank_candidates_for_plan(
            backend,
            judged_query.query,
            snapshot,
            stage1.hits,
            heldout_task.k,
        )
        var oracle_final_hits = exact_oracle_hits_for_snapshot(
            backend,
            judged_query.query,
            snapshot,
            heldout_task.k,
        )
        candidate_recall_total += collection_hit_recall_at_final_k(
            stage1.hits,
            oracle_final_hits,
        )
        var query_evaluation = evaluate_query_hits(
            judged_query,
            final_hits_to_search_hits(stage2.final_hits),
            heldout_task.k,
            heldout_task.primary_metric,
        )
        recall_total += Float64(query_evaluation.recall_at_k)
        success_total += Float64(query_evaluation.success_at_k)
        visited_vertex_total += Float64(
            stage1.graph_search_counters.visited_vertex_count
        )
        expanded_edge_total += Float64(
            stage1.graph_search_counters.expanded_edge_count
        )
        visited_cluster_total += Float64(
            stage1.graph_search_counters.visited_cluster_count
        )
        entry_point_total += Float64(
            stage1.graph_search_counters.entry_point_count
        )
        max_frontier_total += Float64(
            stage1.graph_search_counters.max_frontier_size
        )
    var query_count = len(heldout_task.queries)
    return GemHeldoutFrontierPolicyProbeSummary(
        heldout_task.slice_name.copy(),
        variant_kind,
        frontier_policy,
        candidate_k,
        search_cluster_top_k_per_query_token,
        search_beam_width,
        candidate_recall_total / Float64(query_count),
        recall_total / Float64(query_count),
        success_total / Float64(query_count),
        visited_vertex_total / Float64(query_count),
        expanded_edge_total / Float64(query_count),
        visited_cluster_total / Float64(query_count),
        entry_point_total / Float64(query_count),
        max_frontier_total / Float64(query_count),
    )


def append_gem_heldout_frontier_policy_probe_summary_json(
    mut buffer: String, read summary: GemHeldoutFrontierPolicyProbeSummary
):
    buffer += "{"
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"variant_kind\":\"" + json_escape(summary.variant_kind) + "\","
    buffer += "\"frontier_policy\":\"" + json_escape(summary.frontier_policy) + "\","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"search_cluster_top_k_per_query_token\":"
    buffer += String(summary.search_cluster_top_k_per_query_token) + ","
    buffer += "\"search_beam_width\":"
    buffer += String(summary.search_beam_width) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_recall_at_k\":"
    buffer += String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":"
    buffer += String(summary.success_rate_at_k) + ","
    buffer += "\"mean_stage1_graph_visited_vertex_count\":"
    buffer += String(summary.mean_stage1_graph_visited_vertex_count) + ","
    buffer += "\"mean_stage1_graph_expanded_edge_count\":"
    buffer += String(summary.mean_stage1_graph_expanded_edge_count) + ","
    buffer += "\"mean_stage1_graph_visited_cluster_count\":"
    buffer += String(summary.mean_stage1_graph_visited_cluster_count) + ","
    buffer += "\"mean_stage1_graph_entry_point_count\":"
    buffer += String(summary.mean_stage1_graph_entry_point_count) + ","
    buffer += "\"mean_stage1_graph_max_frontier_size\":"
    buffer += String(summary.mean_stage1_graph_max_frontier_size)
    buffer += "}"


def gem_heldout_frontier_policy_probe_summaries_json(
    read summaries: List[GemHeldoutFrontierPolicyProbeSummary]
) -> String:
    var buffer = String("[")
    for summary_index in range(len(summaries)):
        if summary_index > 0:
            buffer += ","
        append_gem_heldout_frontier_policy_probe_summary_json(
            buffer,
            summaries[summary_index],
        )
    buffer += "]"
    return buffer^


def print_gem_heldout_frontier_policy_probe_summary_line(
    read summary: GemHeldoutFrontierPolicyProbeSummary
):
    print(
        "frontier_policy_probe profile=",
        summary.slice_name,
        " variant=",
        summary.variant_kind,
        " policy=",
        summary.frontier_policy,
        " candidate_k=",
        summary.candidate_k,
        " candidate_recall=",
        summary.mean_candidate_recall_at_final_k,
        " recall=",
        summary.mean_recall_at_k,
        " visited=",
        summary.mean_stage1_graph_visited_vertex_count,
        " frontier=",
        summary.mean_stage1_graph_max_frontier_size,
    )


def synthetic_hard_recall_gem_heldout_frontier_policy_probe_summaries(
) raises -> List[GemHeldoutFrontierPolicyProbeSummary]:
    var profiles = default_synthetic_hard_recall_profiles()
    if len(profiles) == 0:
        raise Error(
            "held-out GEM frontier-policy probe requires at least one synthetic profile"
        )

    var fixture = make_synthetic_hard_recall_fixture(profiles[0])
    var split = build_gem_heldout_query_split(
        fixture.stored_task,
        default_gem_heldout_training_query_count(fixture.stored_task),
    )
    var base_config = frontier_gem_graph_build_config(
        fixture.stored_index,
        fixture.stored_task.task.nominal_query_vector_count,
    )
    var backend = gem_heldout_frontier_policy_probe_backend()
    var summaries = List[GemHeldoutFrontierPolicyProbeSummary]()
    var variants = [
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_BASELINE,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
            base_config,
            split.training_pairs,
        ),
    ]
    for variant in variants:
        var collection_path = materialized_gem_heldout_querytime_probe_collection_root(
            gem_heldout_querytime_probe_collection_root(
                fixture.stored_task.task.slice_name,
                variant.variant_kind,
            ),
            gem_heldout_querytime_probe_collection_id(
                fixture.stored_task.task.slice_name,
                variant.variant_kind,
            ),
            fixture.stored_index,
            variant,
            SnapshotId("snapshot-0001"),
        )
        var snapshot = load_resolved_collection_snapshot(
            collection_path,
            SnapshotId("snapshot-0001"),
        )
        var stored_gem = loaded_segment_stored_gem_graph_index(snapshot.segments[0])
        for frontier_policy in gem_heldout_frontier_policy_probe_frontier_policies():
            for candidate_k in gem_heldout_frontier_policy_probe_candidate_ks():
                var summary = build_gem_heldout_frontier_policy_probe_summary(
                    backend,
                    split,
                    snapshot,
                    stored_gem,
                    variant.variant_kind,
                    frontier_policy,
                    candidate_k,
                    gem_heldout_frontier_policy_probe_search_cluster_top_k(),
                    gem_heldout_frontier_policy_probe_beam_width(),
                )
                print_gem_heldout_frontier_policy_probe_summary_line(summary)
                summaries.append(summary^)
    return summaries^


def write_synthetic_hard_recall_gem_heldout_frontier_policy_probe() raises:
    var output_dir = Path(".cache/kayak")
    makedirs(output_dir, exist_ok=True)
    (
        output_dir / "synthetic_hard_recall_gem_heldout_frontier_policy_probe.json"
    ).write_text(
        gem_heldout_frontier_policy_probe_summaries_json(
            synthetic_hard_recall_gem_heldout_frontier_policy_probe_summaries()
        )
    )
