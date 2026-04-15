"""Defines the public text-encoder protocol for late interaction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kayak_bridge import LateDocuments, LateQuery


@runtime_checkable
class LateTextEncoder(Protocol):
    """Protocol for text encoders that emit late-interaction objects."""

    def encode_query(self, text: str) -> LateQuery:
        """Encode one query string into a late-interaction query."""

    def encode_document_vectors(self, text: str) -> object:
        """Encode one document string into token-level vectors."""

    def encode_documents(
        self,
        doc_ids: object,
        texts: object,
    ) -> LateDocuments:
        """Encode aligned document ids and texts into late documents."""
