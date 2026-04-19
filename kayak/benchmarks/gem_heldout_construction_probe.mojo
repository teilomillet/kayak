# Narrow graph-construction probe for held-out GEM baseline on the first real
# synthetic hard-recall profile.
#
# Owns:
# - a fixed-profile sweep over construction_neighbor_count and degree_limit
# - persistent JSON output over fixed search settings
#
# Does not own:
# - generic held-out ablation surfaces
# - adaptive-label-policy comparisons

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.index import GemGraphBuildConfig
from kayak.runtime import ExactCpuBackend
from kayak.scoring import ExactScoringConfig

from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_ablation_json import (
    GEM_HELDOUT_VARIANT_BASELINE,
    build_gem_heldout_query_split,
    default_gem_heldout_training_query_count,
    gem_heldout_ablation_variant_spec,
)
from .gem_heldout_querytime_probe import (
    GemHeldoutQuerytimeProbeSummary,
    build_gem_heldout_querytime_probe_summary,
    gem_heldout_querytime_probe_summaries_json,
)
from .synthetic_hard_recall_fixture import (
    default_synthetic_hard_recall_profiles,
    make_synthetic_hard_recall_fixture,
)


def gem_heldout_construction_probe_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def gem_heldout_construction_probe_candidate_ks() -> List[Int]:
    return [32, 64]


def gem_heldout_construction_probe_construction_neighbor_counts() -> List[Int]:
    return [4, 6, 12]


def gem_heldout_construction_probe_degree_limits() -> List[Int]:
    return [4, 8, 16]


def gem_heldout_construction_probe_search_cluster_top_k() -> Int:
    return 2


def gem_heldout_construction_probe_beam_width() -> Int:
    return 32


def gem_heldout_construction_probe_collection_root(
    slice_name: String,
    construction_neighbor_count: Int,
    degree_limit: Int,
) -> Path:
    return Path(
        ".cache/kayak/synthetic_hard_recall_gem_heldout_construction_probe/"
        + slice_name
        + "/baseline"
        + "/neighbors"
        + String(construction_neighbor_count)
        + "_degree"
        + String(degree_limit)
    )


def gem_heldout_construction_probe_collection_id(
    slice_name: String,
    construction_neighbor_count: Int,
    degree_limit: Int,
) -> String:
    return (
        "synthetic-hard-recall-gem-heldout-construction-probe-"
        + slice_name
        + "-baseline-neighbors"
        + String(construction_neighbor_count)
        + "-degree"
        + String(degree_limit)
    )


def construction_probe_base_config(
    read base_config: GemGraphBuildConfig,
    construction_neighbor_count: Int,
    degree_limit: Int,
) raises -> GemGraphBuildConfig:
    return GemGraphBuildConfig(
        base_config.fine_cluster_count,
        base_config.coarse_cluster_count,
        base_config.cluster_cutoff,
        construction_neighbor_count,
        degree_limit,
        base_config.enable_adaptive_cluster_cutoff,
        base_config.adaptive_cluster_cutoff_max,
        base_config.adaptive_tree_max_depth,
        base_config.cluster_top_k_per_query_token,
        base_config.enable_shortcuts,
        base_config.shortcut_candidate_k,
        base_config.shortcut_beam_width,
        base_config.training_pairs.copy(),
        base_config.adaptive_label_policy.copy(),
    )


def print_gem_heldout_construction_probe_summary_line(
    read summary: GemHeldoutQuerytimeProbeSummary
):
    print(
        "construction_probe profile=",
        summary.slice_name,
        " neighbors=",
        summary.construction_neighbor_count,
        " degree_limit=",
        summary.degree_limit,
        " candidate_k=",
        summary.candidate_k,
        " recall=",
        summary.mean_recall_at_k,
        " reachable=",
        summary.evaluation_positive_reachable_rate,
    )


def synthetic_hard_recall_gem_heldout_construction_probe_summaries(
) raises -> List[GemHeldoutQuerytimeProbeSummary]:
    var profiles = default_synthetic_hard_recall_profiles()
    if len(profiles) == 0:
        raise Error(
            "held-out GEM construction probe requires at least one synthetic profile"
        )

    var fixture = make_synthetic_hard_recall_fixture(profiles[0])
    var split = build_gem_heldout_query_split(
        fixture.stored_task,
        default_gem_heldout_training_query_count(fixture.stored_task),
    )
    var frontier_config = frontier_gem_graph_build_config(
        fixture.stored_index,
        fixture.stored_task.task.nominal_query_vector_count,
    )

    var summaries = List[GemHeldoutQuerytimeProbeSummary]()
    var backend = gem_heldout_construction_probe_backend()
    for construction_neighbor_count in (
        gem_heldout_construction_probe_construction_neighbor_counts()
    ):
        for degree_limit in gem_heldout_construction_probe_degree_limits():
            var variant = gem_heldout_ablation_variant_spec(
                GEM_HELDOUT_VARIANT_BASELINE,
                construction_probe_base_config(
                    frontier_config,
                    construction_neighbor_count,
                    degree_limit,
                ),
                split.training_pairs,
            )
            for candidate_k in gem_heldout_construction_probe_candidate_ks():
                var summary = build_gem_heldout_querytime_probe_summary(
                    backend,
                    fixture.stored_index,
                    split,
                    gem_heldout_construction_probe_collection_root(
                        fixture.stored_task.task.slice_name,
                        construction_neighbor_count,
                        degree_limit,
                    ),
                    gem_heldout_construction_probe_collection_id(
                        fixture.stored_task.task.slice_name,
                        construction_neighbor_count,
                        degree_limit,
                    ),
                    variant.copy(),
                    candidate_k,
                    gem_heldout_construction_probe_search_cluster_top_k(),
                    gem_heldout_construction_probe_beam_width(),
                )
                print_gem_heldout_construction_probe_summary_line(summary)
                summaries.append(summary^)
    return summaries^


def write_synthetic_hard_recall_gem_heldout_construction_probe() raises:
    var output_dir = Path(".cache/kayak")
    makedirs(output_dir, exist_ok=True)
    (
        output_dir / "synthetic_hard_recall_gem_heldout_construction_probe.json"
    ).write_text(
        gem_heldout_querytime_probe_summaries_json(
            synthetic_hard_recall_gem_heldout_construction_probe_summaries()
        )
    )
