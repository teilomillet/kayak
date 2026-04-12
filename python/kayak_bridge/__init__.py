"""Internal compatibility layer for the Python Kayak package.

This module exists so the monorepo can share implementation code between the
public ``kayak`` package and the internal Mojo-backed adapters.
It is not a stable public import surface; application code should import
from ``kayak`` instead.
"""

from .late_documents import LateDocuments
from .late_index import LateIndex
from .late_ops import (
    documents,
    flat_query_dim128,
    hybrid_flat_dim128_index,
    maxsim,
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
    packed_index,
    query,
    search,
)
from .late_query import LateQuery
from .late_scores import LateScores, SearchHit

__all__ = [
    "LateDocuments",
    "LateIndex",
    "LateQuery",
    "LateScores",
    "SearchHit",
    "MOJO_EXACT_CPU_BACKEND",
    "NUMPY_REFERENCE_BACKEND",
    "documents",
    "flat_query_dim128",
    "hybrid_flat_dim128_index",
    "maxsim",
    "packed_index",
    "query",
    "search",
]
