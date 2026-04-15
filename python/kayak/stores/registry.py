"""Owns the public store registry and factory helpers for the SDK."""

from __future__ import annotations

from collections.abc import Callable

from .directory import DirectoryLateStore
from .lancedb_store import LanceDBLateStore
from .memory import MemoryLateStore


StoreFactory = Callable[..., object]

_STORE_FACTORIES: dict[str, StoreFactory] = {
    "directory": DirectoryLateStore,
    "kayak": DirectoryLateStore,
    "lance": LanceDBLateStore,
    "lancedb": LanceDBLateStore,
    "memory": MemoryLateStore,
}


def register_store(
    kind: str,
    factory: StoreFactory,
    *,
    replace: bool = False,
) -> None:
    normalized_kind = kind.strip().lower()
    if normalized_kind == "":
        raise ValueError("store kind must be non-empty")
    if normalized_kind in _STORE_FACTORIES and not replace:
        raise ValueError(f"store kind is already registered: {normalized_kind}")
    _STORE_FACTORIES[normalized_kind] = factory


def open_store(kind: str, /, **kwargs: object) -> object:
    normalized_kind = kind.strip().lower()
    if normalized_kind not in _STORE_FACTORIES:
        raise ValueError(f"unknown store kind: {normalized_kind}")
    return _STORE_FACTORIES[normalized_kind](**kwargs)
