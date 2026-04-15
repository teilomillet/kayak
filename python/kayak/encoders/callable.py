"""Adapts user-provided Python callables into a public late-text encoder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kayak_bridge import LateDocuments, LateQuery, documents, query


@dataclass(frozen=True, slots=True)
class CallableLateTextEncoder:
    """Wraps user-provided text encoders behind the public SDK contract."""

    query_encoder: Callable[[str], object]
    document_encoder: Callable[[str], object]

    def encode_query(self, text: str) -> LateQuery:
        return query(self.query_encoder(text), text=text)

    def encode_document_vectors(self, text: str) -> object:
        return self.document_encoder(text)

    def encode_documents(
        self,
        doc_ids: Sequence[object],
        texts: Sequence[object],
    ) -> LateDocuments:
        text_rows = tuple(str(text) for text in texts)
        token_vectors = tuple(
            self.document_encoder(text) for text in text_rows
        )
        return documents(
            doc_ids,
            token_vectors,
            texts=text_rows,
        )
