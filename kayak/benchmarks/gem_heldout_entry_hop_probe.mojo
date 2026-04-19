# Entry-hop probe for held-out GEM variants on the first real synthetic
# hard-recall profile.
#
# Owns:
# - a fixed-profile comparison between current cluster entry docs and
#   representative-based entry sets
# - persistent JSON output for positive hop-distance diagnostics
#
# Does not own:
# - query-time recall benchmarking
# - generic held-out ablation surfaces

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_ablation_json import (
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
    GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE,
    GEM_HELDOUT_VARIANT_BASELINE,
    GemHeldoutQuerySplit,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    gem_heldout_ablation_variant_spec,
    query_positive_doc_indices,
    query_positive_min_gated_hop_count,
)
from .json_common import json_escape
from .synthetic_hard_recall_fixture import (
    default_synthetic_hard_recall_profiles,
    make_synthetic_hard_recall_fixture,
)

from kayak.collections import (
    SnapshotId,
    load_resolved_collection_snapshot,
    loaded_segment_stored_gem_graph_index,
)
from kayak.index import (
    GemGraphIndex,
    query_entry_doc_indices,
    query_relevant_cluster_ids,
    query_representative_doc_indices,
)
from kayak.storage import StoredGemGraphIndex

comptime ENTRY_HOP_PROBE_STRATEGY_CLUSTER_ENTRY = "cluster_entry"
comptime ENTRY_HOP_PROBE_STRATEGY_REPRESENTATIVE = "representative"


struct GemHeldoutEntryHopProbeSummary(Copyable):
    var slice_name: String
    var variant_kind: String
    var adaptive_label_policy: String
    var search_cluster_top_k_per_query_token: Int
    var entry_strategy: String
    var representative_depth: Int
    var evaluation_positive_mean_reachable_hop_count: Float64
    var evaluation_positive_within_1_hop_rate: Float64
    var evaluation_positive_within_2_hop_rate: Float64
    var evaluation_positive_within_4_hop_rate: Float64
    var evaluation_positive_reachable_rate: Float64

    def __init__(
        out self,
        var slice_name: String,
        var variant_kind: String,
        var adaptive_label_policy: String,
        search_cluster_top_k_per_query_token: Int,
        var entry_strategy: String,
        representative_depth: Int,
        evaluation_positive_mean_reachable_hop_count: Float64,
        evaluation_positive_within_1_hop_rate: Float64,
        evaluation_positive_within_2_hop_rate: Float64,
        evaluation_positive_within_4_hop_rate: Float64,
        evaluation_positive_reachable_rate: Float64,
    ):
        self.slice_name = slice_name^
        self.variant_kind = variant_kind^
        self.adaptive_label_policy = adaptive_label_policy^
        self.search_cluster_top_k_per_query_token = (
            search_cluster_top_k_per_query_token
        )
        self.entry_strategy = entry_strategy^
        self.representative_depth = representative_depth
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
        self.evaluation_positive_reachable_rate = (
            evaluation_positive_reachable_rate
        )


def gem_heldout_entry_hop_probe_search_cluster_top_k() -> Int:
    return 2


def gem_heldout_entry_hop_probe_representative_depths() -> List[Int]:
    return [1, 2, 4, 8]


def gem_heldout_entry_hop_probe_collection_root(
    slice_name: String, variant_kind: String
) -> Path:
    return Path(
        ".cache/kayak/synthetic_hard_recall_gem_heldout_querytime_probe/"
        + slice_name
        + "/"
        + variant_kind
    )


def gem_heldout_entry_hop_probe_collection_id(
    slice_name: String, variant_kind: String
) -> String:
    return (
        "synthetic-hard-recall-gem-heldout-querytime-probe-"
        + slice_name
        + "-"
        + variant_kind
    )


def entry_hop_probe_entry_doc_indices(
    read index: GemGraphIndex,
    read relevant_cluster_ids: List[Int],
    strategy: String,
    representative_depth: Int,
) raises -> List[Int]:
    if strategy == ENTRY_HOP_PROBE_STRATEGY_CLUSTER_ENTRY:
        return query_entry_doc_indices(index, relevant_cluster_ids)
    if strategy == ENTRY_HOP_PROBE_STRATEGY_REPRESENTATIVE:
        return query_representative_doc_indices(
            index,
            relevant_cluster_ids,
            representative_depth,
        )
    raise Error("unknown held-out GEM entry_hop_probe strategy: " + strategy)


def build_gem_heldout_entry_hop_probe_summary(
    read split: GemHeldoutQuerySplit,
    variant_kind: String,
    read stored_gem: StoredGemGraphIndex,
    entry_strategy: String,
    representative_depth: Int,
    search_cluster_top_k_per_query_token: Int,
) raises -> GemHeldoutEntryHopProbeSummary:
    var queries = split.evaluation_task.task.queries.copy()
    if len(queries) == 0:
        return GemHeldoutEntryHopProbeSummary(
            split.evaluation_task.task.slice_name.copy(),
            variant_kind,
            "",
            search_cluster_top_k_per_query_token,
            entry_strategy,
            representative_depth,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    var hop_total = 0
    var reachable_count = 0
    var within_1_count = 0
    var within_2_count = 0
    var within_4_count = 0
    for judged_query in queries:
        var relevant_clusters = query_relevant_cluster_ids(
            judged_query.query,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
        )
        var entry_doc_indices = entry_hop_probe_entry_doc_indices(
            stored_gem.index,
            relevant_clusters,
            entry_strategy,
            representative_depth,
        )
        var min_hops = query_positive_min_gated_hop_count(
            query_positive_doc_indices(judged_query, stored_gem.index.doc_ids),
            stored_gem.index,
            entry_doc_indices,
            relevant_clusters,
        )
        if min_hops < 0:
            continue
        reachable_count += 1
        hop_total += min_hops
        if min_hops <= 1:
            within_1_count += 1
        if min_hops <= 2:
            within_2_count += 1
        if min_hops <= 4:
            within_4_count += 1
    var mean_hops = 0.0
    if reachable_count > 0:
        mean_hops = Float64(hop_total) / Float64(reachable_count)
    return GemHeldoutEntryHopProbeSummary(
        split.evaluation_task.task.slice_name.copy(),
        variant_kind,
        stored_gem.adaptive_label_policy.copy(),
        search_cluster_top_k_per_query_token,
        entry_strategy,
        representative_depth,
        mean_hops,
        Float64(within_1_count) / Float64(len(queries)),
        Float64(within_2_count) / Float64(len(queries)),
        Float64(within_4_count) / Float64(len(queries)),
        Float64(reachable_count) / Float64(len(queries)),
    )


def append_gem_heldout_entry_hop_probe_summary_json(
    mut buffer: String, read summary: GemHeldoutEntryHopProbeSummary
):
    buffer += "{"
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"variant_kind\":\"" + json_escape(summary.variant_kind) + "\","
    buffer += "\"adaptive_label_policy\":\""
    buffer += json_escape(summary.adaptive_label_policy) + "\","
    buffer += "\"search_cluster_top_k_per_query_token\":"
    buffer += String(summary.search_cluster_top_k_per_query_token) + ","
    buffer += "\"entry_strategy\":\"" + json_escape(summary.entry_strategy) + "\","
    buffer += "\"representative_depth\":"
    buffer += String(summary.representative_depth) + ","
    buffer += "\"evaluation_positive_mean_reachable_hop_count\":"
    buffer += String(summary.evaluation_positive_mean_reachable_hop_count) + ","
    buffer += "\"evaluation_positive_within_1_hop_rate\":"
    buffer += String(summary.evaluation_positive_within_1_hop_rate) + ","
    buffer += "\"evaluation_positive_within_2_hop_rate\":"
    buffer += String(summary.evaluation_positive_within_2_hop_rate) + ","
    buffer += "\"evaluation_positive_within_4_hop_rate\":"
    buffer += String(summary.evaluation_positive_within_4_hop_rate) + ","
    buffer += "\"evaluation_positive_reachable_rate\":"
    buffer += String(summary.evaluation_positive_reachable_rate)
    buffer += "}"


def gem_heldout_entry_hop_probe_summaries_json(
    read summaries: List[GemHeldoutEntryHopProbeSummary]
) -> String:
    var buffer = String("[")
    for summary_index in range(len(summaries)):
        if summary_index > 0:
            buffer += ","
        append_gem_heldout_entry_hop_probe_summary_json(
            buffer,
            summaries[summary_index],
        )
    buffer += "]"
    return buffer^


def print_gem_heldout_entry_hop_probe_summary_line(
    read summary: GemHeldoutEntryHopProbeSummary
):
    print(
        "entry_hop_probe profile=",
        summary.slice_name,
        " variant=",
        summary.variant_kind,
        " strategy=",
        summary.entry_strategy,
        " depth=",
        summary.representative_depth,
        " mean_hops=",
        summary.evaluation_positive_mean_reachable_hop_count,
        " within_4=",
        summary.evaluation_positive_within_4_hop_rate,
        " reachable=",
        summary.evaluation_positive_reachable_rate,
    )


def synthetic_hard_recall_gem_heldout_entry_hop_probe_summaries(
) raises -> List[GemHeldoutEntryHopProbeSummary]:
    var profiles = default_synthetic_hard_recall_profiles()
    if len(profiles) == 0:
        raise Error(
            "held-out GEM entry-hop probe requires at least one synthetic profile"
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

    var summaries = List[GemHeldoutEntryHopProbeSummary]()
    for variant in variants:
        var collection_root = gem_heldout_entry_hop_probe_collection_root(
            fixture.stored_task.task.slice_name,
            variant.variant_kind,
        )
        var snapshot = load_resolved_collection_snapshot(
            collection_root,
            SnapshotId("snapshot-0001"),
        )
        var stored_gem = loaded_segment_stored_gem_graph_index(snapshot.segments[0])
        var cluster_entry_summary = build_gem_heldout_entry_hop_probe_summary(
            split,
            variant.variant_kind,
            stored_gem,
            ENTRY_HOP_PROBE_STRATEGY_CLUSTER_ENTRY,
            1,
            gem_heldout_entry_hop_probe_search_cluster_top_k(),
        )
        print_gem_heldout_entry_hop_probe_summary_line(cluster_entry_summary)
        summaries.append(cluster_entry_summary^)
        for representative_depth in (
            gem_heldout_entry_hop_probe_representative_depths()
        ):
            var representative_summary = build_gem_heldout_entry_hop_probe_summary(
                split,
                variant.variant_kind,
                stored_gem,
                ENTRY_HOP_PROBE_STRATEGY_REPRESENTATIVE,
                representative_depth,
                gem_heldout_entry_hop_probe_search_cluster_top_k(),
            )
            print_gem_heldout_entry_hop_probe_summary_line(representative_summary)
            summaries.append(representative_summary^)
    return summaries^


def write_synthetic_hard_recall_gem_heldout_entry_hop_probe() raises:
    var output_dir = Path(".cache/kayak")
    makedirs(output_dir, exist_ok=True)
    (
        output_dir / "synthetic_hard_recall_gem_heldout_entry_hop_probe.json"
    ).write_text(
        gem_heldout_entry_hop_probe_summaries_json(
            synthetic_hard_recall_gem_heldout_entry_hop_probe_summaries()
        )
    )
