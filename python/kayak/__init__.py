"""Public Python SDK for Kayak late-interaction programming.

Import from ``kayak`` when writing application or research code in Python.
This package owns the stable late-interaction object model, exact operations,
and explicit backend selection for the SDK surface.

It is not the hosted engine surface for collections, snapshots, or service
operations. The sibling ``kayak_bridge`` package remains an internal
implementation layer for this monorepo and is not a supported public import
surface.
"""

from kayak_bridge import (
    LateDocuments,
    LateIndex,
    LateQuery,
    LateScores,
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
    SearchHit,
    documents,
    flat_query_dim128,
    hybrid_flat_dim128_index,
    maxsim,
    packed_index,
    query,
    search,
)

PUBLIC_API = (
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
)

__all__ = [
    *PUBLIC_API,
]
