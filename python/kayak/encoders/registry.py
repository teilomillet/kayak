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


def available_encoder_kinds() -> tuple[str, ...]:
    """Return the registered encoder kind strings accepted by ``open_encoder``."""
    return tuple(sorted(_ENCODER_FACTORIES))


def _unknown_encoder_kind_error(kind: object) -> ValueError | TypeError:
    available = ", ".join(available_encoder_kinds())
    if not isinstance(kind, str):
        return TypeError(
            "encoder kind must be a string; "
            f"got {type(kind).__name__}. "
            f"Available kinds: {available}. "
            'Use kayak.available_encoder_kinds() or kayak.help("Encoders").'
        )
    return ValueError(
        f"unknown encoder kind: {kind.strip().lower()}. "
        f"Available kinds: {available}. "
        'Use kayak.available_encoder_kinds() or kayak.help("Encoders").'
    )


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
      callables or one existing model object with named methods
    - ``"colbert"`` for Hugging Face ColBERT checkpoints

    Parameters
    ----------
    kind:
        Registered encoder kind string.
    **kwargs:
        Passed directly to the selected encoder constructor.

    Callable encoder variants
    -------------------------
    The callable encoder supports both:

    - ``query_encoder=...`` and ``document_encoder=...``
    - ``model=...`` plus ``query_method=...`` and ``document_method=...``

    Example
    -------
    >>> encoder = kayak.open_encoder(
    ...     "callable",
    ...     query_encoder=my_query_encoder,
    ...     document_encoder=my_document_encoder,
    ... )
    >>> encoder = kayak.open_encoder(
    ...     "callable",
    ...     model=my_model,
    ...     query_method="encode_query_tokens",
    ...     document_method="encode_document_tokens",
    ... )
    """
    if not isinstance(kind, str):
        raise _unknown_encoder_kind_error(kind)

    normalized_kind = kind.strip().lower()
    if normalized_kind not in _ENCODER_FACTORIES:
        raise _unknown_encoder_kind_error(normalized_kind)
    if normalized_kind == "callable" and "model" in kwargs:
        if "query_encoder" in kwargs or "document_encoder" in kwargs:
            raise ValueError(
                "open_encoder('callable', model=...) cannot be combined with "
                "query_encoder=... or document_encoder=..."
            )
        model = kwargs.pop("model")
        query_method = str(kwargs.pop("query_method", "encode_query_tokens"))
        document_method = str(
            kwargs.pop("document_method", "encode_document_tokens")
        )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                "open_encoder('callable', model=...) received unexpected "
                f"keyword arguments: {unexpected}"
            )
        return CallableLateTextEncoder.from_model(
            model,
            query_method=query_method,
            document_method=document_method,
        )
    return _ENCODER_FACTORIES[normalized_kind](**kwargs)
