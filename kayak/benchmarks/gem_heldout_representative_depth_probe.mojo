# Narrow representative-depth probe for held-out GEM ablations on the first
# real synthetic hard-recall profile.
#
# Owns:
# - a fixed-profile diagnostic for top-m per-cluster representatives
# - persistent JSON output for entry-depth coverage over selected cluster gates
#
# Does not own:
# - generic held-out ablation surfaces
# - query-time graph traversal benchmarking

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.collections import (
    SnapshotId,
    load_resolved_collection_snapshot,
    loaded_segment_stored_gem_graph_index,
)
from kayak.storage import StoredPackedIndex

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
    evaluation_positive_reachable_rate,
    evaluation_positive_representative_rate,
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


struct GemHeldoutRepresentativeDepthProbeSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var collection_id: String
    var snapshot_id: String
    var variant_kind: String
    var adaptive_label_policy: String
    var search_cluster_top_k_per_query_token: Int
    var representative_depth: Int
    var evaluation_positive_entry_rate: Float64
    var evaluation_positive_representative_rate: Float64
    var evaluation_positive_reachable_rate: Float64

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
        search_cluster_top_k_per_query_token: Int,
        representative_depth: Int,
        evaluation_positive_entry_rate: Float64,
        evaluation_positive_representative_rate: Float64,
        evaluation_positive_reachable_rate: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.variant_kind = variant_kind^
        self.adaptive_label_policy = adaptive_label_policy^
        self.search_cluster_top_k_per_query_token = (
            search_cluster_top_k_per_query_token
        )
        self.representative_depth = representative_depth
        self.evaluation_positive_entry_rate = evaluation_positive_entry_rate
        self.evaluation_positive_representative_rate = (
            evaluation_positive_representative_rate
        )
        self.evaluation_positive_reachable_rate = (
            evaluation_positive_reachable_rate
        )


def representative_depth_probe_summary(
    read stored_index: StoredPackedIndex,
    read split: GemHeldoutQuerySplit,
    collection_root: Path,
    collection_id: String,
    read variant: GemHeldoutAblationVariantSpec,
    search_cluster_top_k_per_query_token: Int,
    representative_depth: Int,
) raises -> GemHeldoutRepresentativeDepthProbeSummary:
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
    var heldout_task = split.evaluation_task.task.copy()

    return GemHeldoutRepresentativeDepthProbeSummary(
        split.evaluation_task.dataset_id.copy(),
        split.evaluation_task.model_name.copy(),
        heldout_task.family.copy(),
        heldout_task.slice_name.copy(),
        snapshot.collection.collection_id.value.copy(),
        snapshot.snapshot.snapshot_id.value.copy(),
        variant.variant_kind.copy(),
        stored_gem.adaptive_label_policy.copy(),
        search_cluster_top_k_per_query_token,
        representative_depth,
        evaluation_positive_entry_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
        ),
        evaluation_positive_representative_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
            representative_depth,
        ),
        evaluation_positive_reachable_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            search_cluster_top_k_per_query_token,
        ),
    )


def representative_depth_probe_search_cluster_top_ks() -> List[Int]:
    return [2, 4]


def representative_depth_probe_depths() -> List[Int]:
    return [1, 2, 4, 8]


def append_representative_depth_probe_summary_json(
    mut buffer: String,
    read summary: GemHeldoutRepresentativeDepthProbeSummary,
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
    buffer += "\"search_cluster_top_k_per_query_token\":"
    buffer += String(summary.search_cluster_top_k_per_query_token) + ","
    buffer += "\"representative_depth\":"
    buffer += String(summary.representative_depth) + ","
    buffer += "\"evaluation_positive_entry_rate\":"
    buffer += String(summary.evaluation_positive_entry_rate) + ","
    buffer += "\"evaluation_positive_representative_rate\":"
    buffer += String(summary.evaluation_positive_representative_rate) + ","
    buffer += "\"evaluation_positive_reachable_rate\":"
    buffer += String(summary.evaluation_positive_reachable_rate)
    buffer += "}"


def representative_depth_probe_summaries_json(
    read summaries: List[GemHeldoutRepresentativeDepthProbeSummary]
) -> String:
    var buffer = String("[")
    for summary_index in range(len(summaries)):
        if summary_index > 0:
            buffer += ","
        append_representative_depth_probe_summary_json(
            buffer,
            summaries[summary_index],
        )
    buffer += "]"
    return buffer^


def print_representative_depth_probe_summary_line(
    read summary: GemHeldoutRepresentativeDepthProbeSummary
):
    print(
        "representative_depth_probe profile=",
        summary.slice_name,
        " variant=",
        summary.variant_kind,
        " cluster_top_k=",
        summary.search_cluster_top_k_per_query_token,
        " representative_depth=",
        summary.representative_depth,
        " representative_rate=",
        summary.evaluation_positive_representative_rate,
        " reachable=",
        summary.evaluation_positive_reachable_rate,
    )


def synthetic_hard_recall_gem_heldout_representative_depth_probe_summaries(
) raises -> List[GemHeldoutRepresentativeDepthProbeSummary]:
    var profiles = default_synthetic_hard_recall_profiles()
    if len(profiles) == 0:
        raise Error(
            "held-out GEM representative-depth probe requires at least one synthetic profile"
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

    var summaries = List[GemHeldoutRepresentativeDepthProbeSummary]()
    for variant in variants:
        for search_cluster_top_k in (
            representative_depth_probe_search_cluster_top_ks()
        ):
            for representative_depth in representative_depth_probe_depths():
                var summary = representative_depth_probe_summary(
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
                    search_cluster_top_k,
                    representative_depth,
                )
                print_representative_depth_probe_summary_line(summary)
                summaries.append(summary^)
    return summaries^


def write_synthetic_hard_recall_gem_heldout_representative_depth_probe() raises:
    var output_dir = Path(".cache/kayak")
    makedirs(output_dir, exist_ok=True)
    (
        output_dir
        / "synthetic_hard_recall_gem_heldout_representative_depth_probe.json"
    ).write_text(
        representative_depth_probe_summaries_json(
            synthetic_hard_recall_gem_heldout_representative_depth_probe_summaries()
        )
    )
