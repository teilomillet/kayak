# Reproducible GEM build defaults for frontier comparisons.
#
# These defaults are intentionally conservative and benchmark-facing:
# - they are not presented as paper-optimal GEM hyperparameters
# - they keep the graph build shape explicit and reproducible across slices

from kayak.index import GemGraphBuildConfig
from kayak.storage import StoredPackedIndex


def integer_sqrt(value: Int) -> Int:
    if value <= 0:
        return 0

    var root = 0
    while (root + 1) * (root + 1) <= value:
        root += 1
    return root


def frontier_gem_graph_build_config(
    read stored_index: StoredPackedIndex,
    query_vector_budget: Int,
) raises -> GemGraphBuildConfig:
    var fine_cluster_count = stored_index.index.vector_dim
    if fine_cluster_count > stored_index.index.total_vector_count:
        fine_cluster_count = stored_index.index.total_vector_count
    if fine_cluster_count <= 0:
        fine_cluster_count = 1

    var coarse_cluster_count = integer_sqrt(stored_index.index.document_count)
    if coarse_cluster_count <= 0:
        coarse_cluster_count = 1
    if coarse_cluster_count > fine_cluster_count:
        coarse_cluster_count = fine_cluster_count

    var cluster_cutoff = query_vector_budget
    if cluster_cutoff <= 0:
        cluster_cutoff = 1
    if cluster_cutoff > coarse_cluster_count:
        cluster_cutoff = coarse_cluster_count

    return GemGraphBuildConfig(
        fine_cluster_count,
        coarse_cluster_count,
        cluster_cutoff,
    )

