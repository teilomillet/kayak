"""Adapts user-provided Python callables into a public late-text encoder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kayak_bridge.api_types import DocIdsInput, DocTextsInput, TokenMatrixInput
from kayak_bridge import LateDocuments, LateQuery, documents, query


@dataclass(frozen=True, slots=True)
class CallableLateTextEncoder:
    """Wrap user-provided Python callables behind the public encoder contract.

    Use this when you already have a model or helper functions that emit
    token-level vectors and you only want Kayak to adapt them into
    ``LateQuery`` and ``LateDocuments`` objects.
    """

    query_encoder: Callable[[str], TokenMatrixInput]
    document_encoder: Callable[[str], TokenMatrixInput]

    def encode_query(self, text: str) -> LateQuery:
        """Encode one query string into ``LateQuery``."""
        return query(self.query_encoder(text), text=text)

    def encode_document_vectors(self, text: str) -> TokenMatrixInput:
        """Encode one document string into token-level vectors."""
        return self.document_encoder(text)

    def encode_documents(
        self,
        doc_ids: DocIdsInput,
        texts: DocTextsInput,
    ) -> LateDocuments:
        """Encode aligned document ids and texts into ``LateDocuments``."""
        text_rows = tuple(str(text) for text in texts)
        token_vectors = tuple(
            self.document_encoder(text) for text in text_rows
        )
        return documents(
            doc_ids,
            token_vectors,
            texts=text_rows,
        )
