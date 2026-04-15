"""Public persistence and materialization stores for the Kayak Python SDK.

This package owns the stable storage contract for Python users:
- persist late-interaction documents
- load exact searchable indexes
- keep storage separate from Kayak's search semantics
"""

from .base import LateStore, LateStoreStats, StoreCapabilities
from .directory import DirectoryLateStore
from .lancedb_store import LanceDBLateStore
from .memory import MemoryLateStore
from .registry import open_store, register_store

__all__ = [
    "DirectoryLateStore",
    "LanceDBLateStore",
    "LateStore",
    "LateStoreStats",
    "MemoryLateStore",
    "StoreCapabilities",
    "open_store",
    "register_store",
]
