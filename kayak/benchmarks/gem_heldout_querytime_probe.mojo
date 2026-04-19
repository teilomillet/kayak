# Fast query-time probe for held-out GEM variants.
#
# Owns:
# - query-time-only held-out summaries that skip benchmark timing loops
# - fixed-profile candidate-window sweeps for the first real synthetic slice
#
# Does not own:
# - general held-out ablation benchmarking
# - latency benchmarking

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    loaded_segment_stored_gem_graph_index,
)
from kayak.eval import evaluate_query_hits
from kayak.planning import (
    best_effort_faithfulness_policy,
    explain_collection_search,
    final_hits_to_search_hits,
    gem_graph_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig
from kayak.storage import StoredGemGraphIndex, StoredPackedIndex

from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_ablation_json import (
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
    GEM_HELDOUT_VARIANT_BASELINE,
    GemHeldoutAblationVariantSpec,
    GemHeldoutQuerySplit,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    evaluation_positive_entry_rate,
    evaluation_positive_mean_reachable_hop_count,
    evaluation_positive_profile_hit_rate,
    evaluation_positive_reachable_rate,
    evaluation_positive_within_hop_rate,
    gem_heldout_ablation_variant_spec,
    mean_document_profile_limit,
)
from .json_common import json_escape
from .query_text_support import judged_query_text_for_plan
from .synthetic_hard_recall_fixture import (
    default_synthetic_hard_recall_profiles,
    make_synthetic_hard_recall_fixture,
)


struct GemHeldoutQuerytimeProbeSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var collection_id: String
    var snapshot_id: String
    var variant_kind: String
    var adaptive_label_policy: String
    var construction_neighbor_count: Int
    var degree_limit: Int
    var shortcut_candidate_k: Int
    var shortcuts_enabled: Bool
    var artifact_shortcut_edge_count: Int
    var candidate_k: Int
    var search_cluster_top_k_per_query_token: Int
    var search_beam_width: Int
    var query_vector_budget: Int
    var artifact_graph_edge_count: Int
    var mean_document_profile_limit: Float64
    var evaluation_positive_profile_hit_rate: Float64
    var evaluation_positive_entry_rate: Float64
    var evaluation_positive_reachable_rate: Float64
    var evaluation_positive_mean_reachable_hop_count: Float64
    var evaluation_positive_within_1_hop_rate: Float64
    var evaluation_positive_within_2_hop_rate: Float64
    var evaluation_positive_within_4_hop_rate: Float64
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
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var collection_id: String,
        var snapshot_id: String,
        var variant_kind: String,
        var adaptive_label_policy: String,
        construction_neighbor_count: Int,
        degree_limit: Int,
        shortcut_candidate_k: Int,
        shortcuts_enabled: Bool,
        artifact_shortcut_edge_count: Int,
        candidate_k: Int,
        search_cluster_top_k_per_query_token: Int,
        search_beam_width: Int,
        query_vector_budget: Int,
        artifact_graph_edge_count: Int,
        mean_document_profile_limit: Float64,
        evaluation_positive_profile_hit_rate: Float64,
        evaluation_positive_entry_rate: Float64,
        evaluation_positive_reachable_rate: Float64,
        evaluation_positive_mean_reachable_hop_count: Float64,
        evaluation_positive_within_1_hop_rate: Float64,
        evaluation_positive_within_2_hop_rate: Float64,
        evaluation_positive_within_4_hop_rate: Float64,
        mean_candidate_recall_at_final_k: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        mean_stage1_graph_visited_vertex_count: Float64,
        mean_stage1_graph_expanded_edge_count: Float64,
        mean_stage1_graph_visited_cluster_count: Float64,
        mean_stage1_graph_entry_point_count: Float64,
        mean_stage1_graph_max_frontier_size: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.variant_kind = variant_kind^
        self.adaptive_label_policy = adaptive_label_policy^
        self.construction_neighbor_count = construction_neighbor_count
        self.degree_limit = degree_limit
        self.shortcut_candidate_k = shortcut_candidate_k
        self.shortcuts_enabled = shortcuts_enabled
        self.artifact_shortcut_edge_count = artifact_shortcut_edge_count
        self.candidate_k = candidate_k
        self.search_cluster_top_k_per_query_token = (
            search_cluster_top_k_per_query_token
        )
        self.search_beam_width = search_beam_width
        self.query_vector_budget = query_vector_budget
        self.artifact_graph_edge_count = artifact_graph_edge_count
        self.mean_document_profile_limit = mean_document_profile_limit
        self.evaluation_positive_profile_hit_rate = (
            evaluation_positive_profile_hit_rate
        )
        self.evaluation_positive_entry_rate = evaluation_positive_entry_rate
        self.evaluation_positive_reachable_rate = (
            evaluation_positive_reachable_rate
        )
        self.evaluation_positive_mean_reachable_hop_count = (
            evaluation_positive_mean_reachable_hop_count
        )
        self.evaluation_positive_within_1_hop_rate = (
            evaluation_positive_within_1_hop_rate
        )
        self.evaluation_positive_within_2_hop_rate = (
            evaluation_positive_within_2_hop_rate
        )
        self.evaluation_positive_within_4_hop_rate = (
            evaluation_positive_within_4_hop_rate
        )
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


def gem_heldout_querytime_probe_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def build_gem_heldout_querytime_probe_summary(
    read backend: ExactCpuBackend,
    read stored_index: StoredPackedIndex,
    read split: GemHeldoutQuerySplit,
    collection_root: Path,
    collection_id: String,
    read variant: GemHeldoutAblationVariantSpec,
    candidate_k: Int,
    search_cluster_top_k_per_query_token: Int,
    search_beam_width: Int,
) raises -> GemHeldoutQuerytimeProbeSummary:
    var query_vector_budget = split.evaluation_task.task.nominal_query_vector_count
    var heldout_task = split.evaluation_task.task.copy()
    var snapshot_id = SnapshotId("snapshot-0001")
    var collection_path = materialized_gem_heldout_querytime_probe_collection_root(
        collection_root,
        collection_id,
        stored_index,
        variant,
        snapshot_id,
    )
    var snapshot = load_resolved_collection_snapshot(collection_path, snapshot_id)
    var stored_gem = loaded_segment_stored_gem_graph_index(snapshot.segments[0])
    var plan = gem_graph_search_plan(
        heldout_task.k,
        candidate_k,
        best_effort_faithfulness_policy(),
        search_cluster_top_k_per_query_token,
        search_beam_width,
    )

    var candidate_recall_total = 0.0
    var recall_total = 0.0
    var success_total = 0.0
    var visited_vertex_total = 0.0
    var expanded_edge_total = 0.0
    var visited_cluster_total = 0.0
    var entry_point_total = 0.0
    var max_frontier_total = 0.0

    for judged_query in heldout_task.queries:
        var explain = explain_collection_search(
            backend,
            judged_query.query,
            snapshot,
            plan,
            query_text=judged_query_text_for_plan(
                plan,
                judged_query.description,
            ),
        )
        var query_evaluation = evaluate_query_hits(
            judged_query,
            final_hits_to_search_hits(explain.final_hits),
            heldout_task.k,
            heldout_task.primary_metric,
        )
        candidate_recall_total += Float64(explain.candidate_recall_at_final_k)
        recall_total += Float64(query_evaluation.recall_at_k)
        success_total += Float64(query_evaluation.success_at_k)
        visited_vertex_total += Float64(
            explain.candidate_stage.graph_search_counters.visited_vertex_count
        )
        expanded_edge_total += Float64(
            explain.candidate_stage.graph_search_counters.expanded_edge_count
        )
        visited_cluster_total += Float64(
            explain.candidate_stage.graph_search_counters.visited_cluster_count
        )
        entry_point_total += Float64(
            explain.candidate_stage.graph_search_counters.entry_point_count
        )
        max_frontier_total += Float64(
            explain.candidate_stage.graph_search_counters.max_frontier_size
        )

    var query_count = len(heldout_task.queries)
    return GemHeldoutQuerytimeProbeSummary(
        split.evaluation_task.dataset_id.copy(),
        split.evaluation_task.model_name.copy(),
        heldout_task.family.copy(),
        heldout_task.slice_name.copy(),
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        variant.variant_kind.copy(),
        stored_gem.adaptive_label_policy.copy(),
        stored_gem.construction_neighbor_count,
        stored_gem.degree_limit,
        stored_gem.shortcut_candidate_k,
        stored_gem.shortcuts_enabled,
        stored_gem.shortcut_edge_count,
        candidate_k,
        search_cluster_top_k_per_query_token,
        search_beam_width,
        query_vector_budget,
        stored_gem.graph_edge_count,
        mean_document_profile_limit(
            stored_gem.index.doc_profile_offsets,
            stored_gem.document_count,
        ),
        evaluation_positive_profile_hit_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index.doc_profile_offsets,
            stored_gem.index.doc_profile_cluster_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
        ),
        evaluation_positive_entry_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
        ),
        evaluation_positive_reachable_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
        ),
        evaluation_positive_mean_reachable_hop_count(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
        ),
        evaluation_positive_within_hop_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
            1,
        ),
        evaluation_positive_within_hop_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
            2,
        ),
        evaluation_positive_within_hop_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
            4,
        ),
        candidate_recall_total / Float64(query_count),
        recall_total / Float64(query_count),
        success_total / Float64(query_count),
        visited_vertex_total / Float64(query_count),
        expanded_edge_total / Float64(query_count),
        visited_cluster_total / Float64(query_count),
        entry_point_total / Float64(query_count),
        max_frontier_total / Float64(query_count),
    )


def gem_heldout_querytime_probe_collection_manifest_path(root: Path) -> Path:
    return root / "collection.manifest.tsv"


def materialized_gem_heldout_querytime_probe_collection_root(
    collection_root: Path,
    collection_id: String,
    read stored_index: StoredPackedIndex,
    read variant: GemHeldoutAblationVariantSpec,
    snapshot_id: SnapshotId,
) raises -> Path:
    if gem_heldout_querytime_probe_collection_manifest_path(collection_root).exists():
        var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)
        var stored_gem = loaded_segment_stored_gem_graph_index(snapshot.segments[0])
        if stored_gem_graph_matches_probe_variant(
            stored_gem,
            variant.build_config,
        ):
            return collection_root

    # The probe prefers prebuilt caches when available, but tests and cold-cache
    # runs still need a self-contained materialization path. Rebuild when stored
    # GEM metadata drifts from the requested variant so probe summaries do not
    # silently report stale graph-construction parameters.
    return ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId(collection_id),
        TenantId("public"),
        NamespaceId("benchmark"),
        snapshot_id,
        1,
        stored_index,
        variant.build_config,
    )


def stored_gem_graph_matches_probe_variant(
    read stored_gem: StoredGemGraphIndex, read build_config: GemGraphBuildConfig
) -> Bool:
    if stored_gem.cluster_cutoff != build_config.cluster_cutoff:
        return False
    if (
        stored_gem.adaptive_cluster_cutoff_enabled
        != build_config.enable_adaptive_cluster_cutoff
    ):
        return False
    var expected_adaptive_max = 0
    if build_config.enable_adaptive_cluster_cutoff:
        expected_adaptive_max = build_config.adaptive_cluster_cutoff_max
    if stored_gem.adaptive_cluster_cutoff_max != expected_adaptive_max:
        return False
    if (
        stored_gem.construction_neighbor_count
        != build_config.construction_neighbor_count
    ):
        return False
    if stored_gem.degree_limit != build_config.degree_limit:
        return False
    if stored_gem.shortcuts_enabled != build_config.enable_shortcuts:
        return False
    if stored_gem.shortcut_candidate_k != build_config.shortcut_candidate_k:
        return False
    if stored_gem.adaptive_label_policy != build_config.adaptive_label_policy:
        return False
    return True


def default_gem_heldout_querytime_probe_collection_root(
    slice_name: String, variant_kind: String
) -> Path:
    return Path(
        ".cache/kayak/synthetic_hard_recall_gem_heldout_querytime_probe/"
        + slice_name
        + "/"
        + variant_kind
    )


def existing_gem_heldout_candidate_probe_collection_root(
    slice_name: String, variant_kind: String
) -> Path:
    return Path(
        ".cache/kayak/synthetic_hard_recall_gem_heldout_candidate_probe/"
        + slice_name
        + "/"
        + variant_kind
    )


def existing_gem_heldout_candidate_probe_quick_collection_root(
    slice_name: String, variant_kind: String
) -> Path:
    return Path(
        ".cache/kayak/synthetic_hard_recall_gem_heldout_candidate_probe_quick/"
        + slice_name
        + "/"
        + variant_kind
    )


def existing_gem_heldout_beam_probe_collection_root(
    slice_name: String, variant_kind: String
) -> Path:
    return Path(
        ".cache/kayak/synthetic_hard_recall_gem_heldout_beam_probe/"
        + slice_name
        + "/"
        + variant_kind
        + "/beam32"
    )


def gem_heldout_querytime_probe_collection_root(
    slice_name: String, variant_kind: String
) -> Path:
    var candidate_root = existing_gem_heldout_candidate_probe_collection_root(
        slice_name,
        variant_kind,
    )
    if gem_heldout_querytime_probe_collection_manifest_path(candidate_root).exists():
        return candidate_root

    var quick_root = existing_gem_heldout_candidate_probe_quick_collection_root(
        slice_name,
        variant_kind,
    )
    if gem_heldout_querytime_probe_collection_manifest_path(quick_root).exists():
        return quick_root

    var beam_root = existing_gem_heldout_beam_probe_collection_root(
        slice_name,
        variant_kind,
    )
    if gem_heldout_querytime_probe_collection_manifest_path(beam_root).exists():
        return beam_root

    var default_root = default_gem_heldout_querytime_probe_collection_root(
        slice_name,
        variant_kind,
    )
    if gem_heldout_querytime_probe_collection_manifest_path(default_root).exists():
        return default_root

    return default_root


def gem_heldout_querytime_probe_collection_id(
    slice_name: String, variant_kind: String
) -> String:
    return (
        "synthetic-hard-recall-gem-heldout-querytime-probe-"
        + slice_name
        + "-"
        + variant_kind
    )


def append_gem_heldout_querytime_probe_summary_json(
    mut buffer: String, read summary: GemHeldoutQuerytimeProbeSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"variant_kind\":\"" + json_escape(summary.variant_kind) + "\","
    buffer += "\"adaptive_label_policy\":\""
    buffer += json_escape(summary.adaptive_label_policy) + "\","
    buffer += "\"construction_neighbor_count\":"
    buffer += String(summary.construction_neighbor_count) + ","
    buffer += "\"degree_limit\":"
    buffer += String(summary.degree_limit) + ","
    buffer += "\"shortcut_candidate_k\":"
    buffer += String(summary.shortcut_candidate_k) + ","
    buffer += "\"shortcuts_enabled\":"
    if summary.shortcuts_enabled:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"artifact_shortcut_edge_count\":"
    buffer += String(summary.artifact_shortcut_edge_count) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"search_cluster_top_k_per_query_token\":"
    buffer += String(summary.search_cluster_top_k_per_query_token) + ","
    buffer += "\"search_beam_width\":" + String(summary.search_beam_width) + ","
    buffer += "\"query_vector_budget\":" + String(summary.query_vector_budget) + ","
    buffer += "\"artifact_graph_edge_count\":"
    buffer += String(summary.artifact_graph_edge_count) + ","
    buffer += "\"mean_document_profile_limit\":"
    buffer += String(summary.mean_document_profile_limit) + ","
    buffer += "\"evaluation_positive_profile_hit_rate\":"
    buffer += String(summary.evaluation_positive_profile_hit_rate) + ","
    buffer += "\"evaluation_positive_entry_rate\":"
    buffer += String(summary.evaluation_positive_entry_rate) + ","
    buffer += "\"evaluation_positive_reachable_rate\":"
    buffer += String(summary.evaluation_positive_reachable_rate) + ","
    buffer += "\"evaluation_positive_mean_reachable_hop_count\":"
    buffer += String(summary.evaluation_positive_mean_reachable_hop_count) + ","
    buffer += "\"evaluation_positive_within_1_hop_rate\":"
    buffer += String(summary.evaluation_positive_within_1_hop_rate) + ","
    buffer += "\"evaluation_positive_within_2_hop_rate\":"
    buffer += String(summary.evaluation_positive_within_2_hop_rate) + ","
    buffer += "\"evaluation_positive_within_4_hop_rate\":"
    buffer += String(summary.evaluation_positive_within_4_hop_rate) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
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


def gem_heldout_querytime_probe_summaries_json(
    read summaries: List[GemHeldoutQuerytimeProbeSummary]
) -> String:
    var buffer = String("[")
    for summary_index in range(len(summaries)):
        if summary_index > 0:
            buffer += ","
        append_gem_heldout_querytime_probe_summary_json(
            buffer,
            summaries[summary_index],
        )
    buffer += "]"
    return buffer^


def print_gem_heldout_querytime_probe_summary_line(
    read summary: GemHeldoutQuerytimeProbeSummary
):
    print(
        "querytime_probe profile=",
        summary.slice_name,
        " variant=",
        summary.variant_kind,
        " build_neighbors=",
        summary.construction_neighbor_count,
        " degree_limit=",
        summary.degree_limit,
        " shortcut_candidate_k=",
        summary.shortcut_candidate_k,
        " shortcuts=",
        summary.artifact_shortcut_edge_count,
        " candidate_k=",
        summary.candidate_k,
        " cluster_top_k=",
        summary.search_cluster_top_k_per_query_token,
        " beam=",
        summary.search_beam_width,
        " recall=",
        summary.mean_recall_at_k,
        " mean_hops=",
        summary.evaluation_positive_mean_reachable_hop_count,
        " reachable=",
        summary.evaluation_positive_reachable_rate,
    )


def write_synthetic_hard_recall_gem_heldout_querytime_probe() raises:
    var profiles = default_synthetic_hard_recall_profiles()
    var fixture = make_synthetic_hard_recall_fixture(profiles[0])
    var split = build_gem_heldout_query_split(
        fixture.stored_task,
        default_gem_heldout_training_query_count(fixture.stored_task),
    )
    var base_config = frontier_gem_graph_build_config(
        fixture.stored_index,
        fixture.stored_task.task.nominal_query_vector_count,
    )
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

    var summaries = List[GemHeldoutQuerytimeProbeSummary]()
    var backend = gem_heldout_querytime_probe_backend()
    for variant in variants:
        for candidate_k in [4, 8, 16, 32, 64]:
            var summary = build_gem_heldout_querytime_probe_summary(
                backend,
                fixture.stored_index,
                split,
                gem_heldout_querytime_probe_collection_root(
                    fixture.stored_task.task.slice_name,
                    variant.variant_kind,
                ),
                gem_heldout_querytime_probe_collection_id(
                    fixture.stored_task.task.slice_name,
                    variant.variant_kind,
                ),
                variant.copy(),
                candidate_k,
                variant.build_config.cluster_top_k_per_query_token,
                32,
            )
            print_gem_heldout_querytime_probe_summary_line(summary)
            summaries.append(summary^)
    var output_dir = Path(".cache/kayak")
    makedirs(output_dir, exist_ok=True)
    (output_dir / "synthetic_hard_recall_gem_heldout_querytime_probe.json").write_text(
        gem_heldout_querytime_probe_summaries_json(summaries)
    )
