"""Compatibility import surface for the Tachiom TAC first-gate probe.

The implementation is split by responsibility:
- `tachiom_types`: public config/report dataclasses
- `tachiom_allocation`: token-aware centroid budget allocation
- `tachiom_index`: prepared index and exact rerank orchestration
- `tachiom_pq`: residual-PQ candidate-window refine reference
- `tachiom_metrics`: reference search and recall metrics
"""

from __future__ import annotations

from .tachiom_allocation import allocate_tac_centroid_counts
from .tachiom_full import TachiomTacHnswResidualPqIndex
from .tachiom_index import TachiomTacIndex
from .tachiom_metrics import (
    exact_search_positions,
    mean_candidate_set_recall_at_k,
    mean_recall_at_k,
)
from .tachiom_mojo import (
    TachiomResidualPqMojoIndex,
    TachiomTacHnswMojoIndex,
    TachiomTacI8MojoIndex,
    TachiomTacMojoIndex,
)
from .tachiom_hnsw import TachiomHnswConfig, TachiomTacHnswIndex
from .tachiom_pq import TachiomResidualPqConfig, TachiomResidualPqIndex
from .tachiom_types import TachiomTacConfig, TacAllocationSummary

__all__ = [
    "TachiomTacConfig",
    "TacAllocationSummary",
    "TachiomTacIndex",
    "TachiomTacMojoIndex",
    "TachiomTacI8MojoIndex",
    "TachiomHnswConfig",
    "TachiomTacHnswIndex",
    "TachiomTacHnswMojoIndex",
    "TachiomResidualPqConfig",
    "TachiomResidualPqIndex",
    "TachiomResidualPqMojoIndex",
    "TachiomTacHnswResidualPqIndex",
    "allocate_tac_centroid_counts",
    "exact_search_positions",
    "mean_candidate_set_recall_at_k",
    "mean_recall_at_k",
]
