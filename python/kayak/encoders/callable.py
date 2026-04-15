"""Adapts user-provided Python callables into a public late-text encoder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kayak_bridge.api_types import DocIdsInput, DocTextsInput, TokenMatrixInput
from kayak_bridge import LateDocuments, LateQuery, documents, query


def _bound_model_method(
    model: object,
    *,
    method_name: str,
    role: str,
) -> Callable[[str], TokenMatrixInput]:
    method = getattr(model, method_name, None)
    if method is None:
        available = sorted(
            name
            for name in dir(model)
            if not name.startswith("_") and callable(getattr(model, name, None))
        )
        raise ValueError(
            f"model does not define {role} method {method_name!r}; "
            f"available public callables: {', '.join(available) or '(none)'}"
        )
    if not callable(method):
        raise ValueError(
            f"model attribute {method_name!r} exists but is not callable"
        )
    return method


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

    @classmethod
    def from_model(
        cls,
        model: object,
        *,
        query_method: str = "encode_query_tokens",
        document_method: str = "encode_document_tokens",
    ) -> "CallableLateTextEncoder":
        """Wrap one model object with named query and document methods.

        Use this when you already have one model instance and want Kayak to
        bind its query/document vector methods into the public encoder
        contract without writing small wrapper lambdas yourself.

        Parameters
        ----------
        model:
            User model instance that exposes one query method and one document
            method.
        query_method:
            Method name called as ``model.<query_method>(text)`` for queries.
        document_method:
            Method name called as ``model.<document_method>(text)`` for
            documents.
        """
        return cls(
            query_encoder=_bound_model_method(
                model,
                method_name=query_method,
                role="query",
            ),
            document_encoder=_bound_model_method(
                model,
                method_name=document_method,
                role="document",
            ),
        )

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
