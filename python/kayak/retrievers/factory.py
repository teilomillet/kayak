"""Owns convenience construction for the public text retriever."""

from __future__ import annotations

from ..encoders import LateTextEncoder, open_encoder
from ..stores import LateStore, open_store
from .text import LateTextRetriever


def _resolve_encoder(
    encoder: str | LateTextEncoder | object,
    *,
    encoder_kwargs: dict[str, object] | None,
) -> LateTextEncoder:
    kwargs = dict(encoder_kwargs or {})
    if isinstance(encoder, str):
        return open_encoder(encoder, **kwargs)
    if isinstance(encoder, LateTextEncoder):
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                "encoder_kwargs are only supported when encoder is a "
                "registered kind string or a model object; got keyword "
                f"arguments for an already-constructed encoder: {unexpected}"
            )
        return encoder
    return open_encoder("callable", model=encoder, **kwargs)


def _resolve_store(
    store: str | LateStore | object,
    *,
    store_kwargs: dict[str, object] | None,
) -> LateStore:
    kwargs = dict(store_kwargs or {})
    if isinstance(store, str):
        return open_store(store, **kwargs)
    if not isinstance(store, LateStore):
        raise TypeError(
            "store must be a registered kind string or one object implementing "
            "the public LateStore protocol "
            "(capabilities, stats, upsert, delete, load_index, close, "
            "__enter__, __exit__). "
            'Use kayak.open_store(...) to construct a built-in store.'
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "store_kwargs are only supported when store is a registered kind "
            "string; got keyword arguments for an already-constructed store: "
            f"{unexpected}"
        )
    return store


def open_text_retriever(
    *,
    encoder: str | LateTextEncoder | object,
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
    exact or staged search. For repeated traffic against one stable slice, use
    ``retriever.session(...)`` after construction.

    Parameters
    ----------
    encoder:
        Either a registered encoder kind such as ``"callable"`` or
        ``"colbert"``, an already-constructed encoder object, or one model
        object that exposes ``encode_query_tokens(text)`` and
        ``encode_document_tokens(text)``.
    store:
        Either a registered store kind such as ``"memory"``, ``"kayak"``,
        ``"lancedb"``, ``"pgvector"``, ``"qdrant"``, ``"weaviate"``, or
        ``"chromadb"``, or an already-constructed store object.
    backend:
        Optional default backend for exact scoring calls made through this
        retriever. When omitted, Kayak picks the default text-retriever
        backend for the current runtime.
    encoder_kwargs:
        Keyword arguments passed when ``encoder`` is a registered kind string.
        When ``encoder`` is a model object instead of a ready-made encoder,
        these kwargs may provide ``query_method=...`` and
        ``document_method=...``.
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
    >>> retriever = kayak.open_text_retriever(
    ...     encoder=my_model,
    ...     store="memory",
    ... )
    """
    encoder_object = _resolve_encoder(
        encoder,
        encoder_kwargs=encoder_kwargs,
    )
    store_object = _resolve_store(
        store,
        store_kwargs=store_kwargs,
    )
    return LateTextRetriever(
        encoder=encoder_object,
        store=store_object,
        default_backend=backend,
        default_layout=default_layout,
        default_include_text=default_include_text,
    )
