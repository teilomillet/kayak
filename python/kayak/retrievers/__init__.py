"""High-level retrieval workflows for the Kayak Python SDK.

This package owns two compositional retrieval surfaces:
- ``LateTextRetriever`` for ingest plus search over a store-backed workflow
- ``LateTextSearchSession`` for repeated search over one fixed loaded index

It does not own low-level scoring kernels or store implementations.
"""

from .session import LateTextSearchSession
from .text import LateTextRetriever
from .factory import open_text_retriever

__all__ = [
    "LateTextSearchSession",
    "LateTextRetriever",
    "open_text_retriever",
]
