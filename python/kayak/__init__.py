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
    BackendInfo,
    LateDocuments,
    LateIndex,
    LateQuery,
    LateQueryBatch,
    LateScores,
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
    SearchHit,
    available_backends,
    backend_info,
    documents,
    flat_query_dim128,
    hybrid_flat_dim128_index,
    maxsim,
    maxsim_batch,
    packed_index,
    query,
    query_batch,
    search,
    search_batch,
)

PUBLIC_API = (
    "BackendInfo",
    "LateDocuments",
    "LateIndex",
    "LateQuery",
    "LateQueryBatch",
    "LateScores",
    "SearchHit",
    "MOJO_EXACT_CPU_BACKEND",
    "NUMPY_REFERENCE_BACKEND",
    "available_backends",
    "backend_info",
    "documents",
    "flat_query_dim128",
    "hybrid_flat_dim128_index",
    "maxsim",
    "maxsim_batch",
    "packed_index",
    "query",
    "query_batch",
    "search",
    "search_batch",
)

__all__ = [
    *PUBLIC_API,
]
