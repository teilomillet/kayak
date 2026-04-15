"""Owns convenience construction for the public text retriever."""

from __future__ import annotations

from .text import LateTextRetriever
from ..encoders import open_encoder
from ..stores import open_store


def open_text_retriever(
    *,
    encoder: object,
    store: object,
    backend: str | None = None,
    encoder_kwargs: dict[str, object] | None = None,
    store_kwargs: dict[str, object] | None = None,
    default_layout: str = "packed",
    default_include_text: bool = False,
) -> LateTextRetriever:
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
