"""Owns convenience construction for the public text retriever."""

from __future__ import annotations

from ..encoders import LateTextEncoder, open_encoder
from ..stores import LateStore, open_store
from .text import LateTextRetriever


def open_text_retriever(
    *,
    encoder: str | LateTextEncoder,
    store: str | LateStore,
    backend: str | None = None,
    encoder_kwargs: dict[str, object] | None = None,
    store_kwargs: dict[str, object] | None = None,
    default_layout: str = "packed",
    default_include_text: bool = False,
) -> LateTextRetriever:
    """Open one high-level text retriever from an encoder and a store.

    Use this when you want one object that owns the common text workflow:
    encode query text, persist encoded documents, load exact indexes, and run
    exact or staged search.

    Parameters
    ----------
    encoder:
        Either a registered encoder kind such as ``"callable"`` or
        ``"colbert"``, or an already-constructed encoder object.
    store:
        Either a registered store kind such as ``"memory"``, ``"kayak"``,
        ``"lancedb"``, ``"pgvector"``, ``"qdrant"``, ``"weaviate"``, or
        ``"chromadb"``, or an already-constructed store object.
    backend:
        Optional default backend for exact scoring calls made through this
        retriever. When omitted, Kayak picks the default text-retriever
        backend for the current runtime.
    encoder_kwargs:
        Keyword arguments passed only when ``encoder`` is a registered kind
        string.
    store_kwargs:
        Keyword arguments passed only when ``store`` is a registered kind
        string.
    default_layout:
        Default index layout to materialize when loading exact slices.
    default_include_text:
        Whether loaded indexes should include document text by default.

    Returns
    -------
    LateTextRetriever
        One configured text retriever ready for ingest and search.

    Example
    -------
    >>> retriever = kayak.open_text_retriever(
    ...     encoder="callable",
    ...     store="memory",
    ...     encoder_kwargs={
    ...         "query_encoder": my_query_encoder,
    ...         "document_encoder": my_document_encoder,
    ...     },
    ... )
    >>> retriever.upsert_texts(["doc-1"], ["late interaction in python"])
    >>> retriever.search_text("late interaction", k=1)
    """
    encoder_object = (
        open_encoder(encoder, **(encoder_kwargs or {}))
        if isinstance(encoder, str)
        else encoder
    )
    store_object = (
        open_store(store, **(store_kwargs or {}))
        if isinstance(store, str)
        else store
    )
    return LateTextRetriever(
        encoder=encoder_object,
        store=store_object,
        default_backend=backend,
        default_layout=default_layout,
        default_include_text=default_include_text,
    )
