"""Owns the public store registry and factory helpers for the SDK."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

from .base import LateStore
from .chromadb_store import ChromaLateStore
from .directory import DirectoryLateStore
from .lancedb_store import LanceDBLateStore
from .memory import MemoryLateStore
from .pgvector_store import PgVectorLateStore
from .qdrant_store import QdrantLateStore
from .weaviate_store import WeaviateLateStore


StoreFactory = Callable[..., LateStore]

_STORE_FACTORIES: dict[str, StoreFactory] = {
    "chroma": ChromaLateStore,
    "chromadb": ChromaLateStore,
    "directory": DirectoryLateStore,
    "kayak": DirectoryLateStore,
    "lance": LanceDBLateStore,
    "lancedb": LanceDBLateStore,
    "memory": MemoryLateStore,
    "pgvector": PgVectorLateStore,
    "postgres": PgVectorLateStore,
    "postgresql": PgVectorLateStore,
    "qdrant": QdrantLateStore,
    "weaviate": WeaviateLateStore,
}


def register_store(
    kind: str,
    factory: StoreFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one store factory behind ``open_store(...)``.

    Use this when you want your own persistence adapter to be available
    through the stable ``open_store(...)`` entry point.
    """
    normalized_kind = kind.strip().lower()
    if normalized_kind == "":
        raise ValueError("store kind must be non-empty")
    if normalized_kind in _STORE_FACTORIES and not replace:
        raise ValueError(f"store kind is already registered: {normalized_kind}")
    _STORE_FACTORIES[normalized_kind] = factory


@overload
def open_store(
    kind: Literal["memory"],
    /,
    **kwargs: object,
) -> MemoryLateStore: ...


@overload
def open_store(
    kind: Literal["directory", "kayak"],
    /,
    **kwargs: object,
) -> DirectoryLateStore: ...


@overload
def open_store(
    kind: Literal["lance", "lancedb"],
    /,
    **kwargs: object,
) -> LanceDBLateStore: ...


@overload
def open_store(
    kind: Literal["pgvector", "postgres", "postgresql"],
    /,
    **kwargs: object,
) -> PgVectorLateStore: ...


@overload
def open_store(
    kind: Literal["qdrant"],
    /,
    **kwargs: object,
) -> QdrantLateStore: ...


@overload
def open_store(
    kind: Literal["weaviate"],
    /,
    **kwargs: object,
) -> WeaviateLateStore: ...


@overload
def open_store(
    kind: Literal["chroma", "chromadb"],
    /,
    **kwargs: object,
) -> ChromaLateStore: ...


@overload
def open_store(kind: str, /, **kwargs: object) -> LateStore: ...


def open_store(kind: str, /, **kwargs: object) -> LateStore:
    """Open one public late-store implementation by registered kind.

    Built-in kinds:
    - ``"memory"``
    - ``"kayak"`` and ``"directory"``
    - ``"lancedb"`` and ``"lance"``
    - ``"pgvector"``, ``"postgres"``, and ``"postgresql"``
    - ``"qdrant"``
    - ``"weaviate"``
    - ``"chromadb"`` and ``"chroma"``

    Parameters
    ----------
    kind:
        Registered store kind string.
    **kwargs:
        Passed directly to the selected store constructor.

    Example
    -------
    >>> store = kayak.open_store("memory")
    >>> store = kayak.open_store("kayak", path="./kayak-index")
    """
    normalized_kind = kind.strip().lower()
    if normalized_kind not in _STORE_FACTORIES:
        raise ValueError(f"unknown store kind: {normalized_kind}")
    return _STORE_FACTORIES[normalized_kind](**kwargs)
