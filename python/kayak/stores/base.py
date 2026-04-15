"""Defines the public late-store protocol and introspection dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kayak_bridge import LateDocuments, LateIndex


@dataclass(frozen=True, slots=True)
class StoreCapabilities:
    """Describes the supported operations for one late-store implementation."""

    kind: str
    persistent: bool
    supports_metadata_filter: bool
    supports_document_subset_load: bool
    supported_layouts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LateStoreStats:
    """Reports measurable state about one late-store instance."""

    kind: str
    document_count: int
    total_vector_count: int
    vector_dim: int | None
    has_texts: bool
    has_metadata: bool
    storage_byte_size: int | None


@runtime_checkable
class LateStore(Protocol):
    """Protocol for persistence layers that materialize Kayak indexes."""

    def capabilities(self) -> StoreCapabilities:
        """Return the supported behaviors for this store."""

    def stats(self) -> LateStoreStats:
        """Return measurable store state for validation or benchmarking."""

    def upsert(
        self,
        documents: LateDocuments,
        *,
        metadata: object | None = None,
    ) -> None:
        """Insert or replace aligned late-interaction documents."""

    def delete(self, doc_ids: object) -> None:
        """Delete the requested document ids from the store."""

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool = False,
        layout: str = "packed",
    ) -> LateIndex:
        """Materialize one late-interaction index from persisted documents."""
