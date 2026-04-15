"""Owns the public encoder registry and factory helpers for the SDK."""

from __future__ import annotations

from collections.abc import Callable

from .callable import CallableLateTextEncoder
from .colbert import ColBERTTextEncoder


EncoderFactory = Callable[..., object]

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
    normalized_kind = kind.strip().lower()
    if normalized_kind == "":
        raise ValueError("encoder kind must be non-empty")
    if normalized_kind in _ENCODER_FACTORIES and not replace:
        raise ValueError(f"encoder kind is already registered: {normalized_kind}")
    _ENCODER_FACTORIES[normalized_kind] = factory


def open_encoder(kind: str, /, **kwargs: object) -> object:
    normalized_kind = kind.strip().lower()
    if normalized_kind not in _ENCODER_FACTORIES:
        raise ValueError(f"unknown encoder kind: {normalized_kind}")
    return _ENCODER_FACTORIES[normalized_kind](**kwargs)
