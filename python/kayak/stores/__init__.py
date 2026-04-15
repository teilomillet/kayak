"""Public persistence and materialization stores for the Kayak Python SDK.

This package owns the stable storage contract for Python users:
- persist late-interaction documents
- load exact searchable indexes
- keep storage separate from Kayak's search semantics
"""

from .base import LateStore, LateStoreStats, StoreCapabilities
from .chromadb_store import ChromaLateStore
from .directory import DirectoryLateStore
from .lancedb_store import LanceDBLateStore
from .memory import MemoryLateStore
from .pgvector_store import PgVectorLateStore
from .qdrant_store import QdrantLateStore
from .registry import available_store_kinds, open_store, register_store
from .weaviate_store import WeaviateLateStore

__all__ = [
    "available_store_kinds",
    "ChromaLateStore",
    "DirectoryLateStore",
    "LanceDBLateStore",
    "LateStore",
    "LateStoreStats",
    "MemoryLateStore",
    "PgVectorLateStore",
    "QdrantLateStore",
    "StoreCapabilities",
    "WeaviateLateStore",
    "open_store",
    "register_store",
]
