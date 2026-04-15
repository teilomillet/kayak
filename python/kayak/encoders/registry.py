"""Owns the public encoder registry and factory helpers for the SDK."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

from .base import LateTextEncoder
from .callable import CallableLateTextEncoder
from .colbert import ColBERTTextEncoder


EncoderFactory = Callable[..., LateTextEncoder]

_ENCODER_FACTORIES: dict[str, EncoderFactory] = {
    "callable": CallableLateTextEncoder,
    "colbert": ColBERTTextEncoder,
}


def register_encoder(
    kind: str,
    factory: EncoderFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one encoder factory behind ``open_encoder(...)``.

    Use this to expose your own encoder kind through the stable factory
    surface instead of constructing it manually at every call site.
    """
    normalized_kind = kind.strip().lower()
    if normalized_kind == "":
        raise ValueError("encoder kind must be non-empty")
    if normalized_kind in _ENCODER_FACTORIES and not replace:
        raise ValueError(f"encoder kind is already registered: {normalized_kind}")
    _ENCODER_FACTORIES[normalized_kind] = factory


@overload
def open_encoder(
    kind: Literal["callable"],
    /,
    **kwargs: object,
) -> CallableLateTextEncoder: ...


@overload
def open_encoder(
    kind: Literal["colbert"],
    /,
    **kwargs: object,
) -> ColBERTTextEncoder: ...


@overload
def open_encoder(kind: str, /, **kwargs: object) -> LateTextEncoder: ...


def open_encoder(kind: str, /, **kwargs: object) -> LateTextEncoder:
    """Open one public text encoder by registered kind.

    Built-in kinds:
    - ``"callable"`` for wrapping your own query and document encoder
      callables
    - ``"colbert"`` for Hugging Face ColBERT checkpoints

    Parameters
    ----------
    kind:
        Registered encoder kind string.
    **kwargs:
        Passed directly to the selected encoder constructor.

    Example
    -------
    >>> encoder = kayak.open_encoder(
    ...     "callable",
    ...     query_encoder=my_query_encoder,
    ...     document_encoder=my_document_encoder,
    ... )
    """
    normalized_kind = kind.strip().lower()
    if normalized_kind not in _ENCODER_FACTORIES:
        raise ValueError(f"unknown encoder kind: {normalized_kind}")
    return _ENCODER_FACTORIES[normalized_kind](**kwargs)
