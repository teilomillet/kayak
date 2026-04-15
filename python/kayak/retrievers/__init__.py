"""High-level retrieval workflows for the Kayak Python SDK.

This package owns one compositional text retrieval surface:
- pair a text encoder with a late store
- keep the encoder and store injectable
- provide one object for ingest, load, and search flows

It does not own low-level scoring kernels or store implementations.
"""

from .text import LateTextRetriever
from .factory import open_text_retriever

__all__ = [
    "LateTextRetriever",
    "open_text_retriever",
]
