"""Public text-to-late-interaction encoders for the Kayak Python SDK.

This package owns the stable Python-facing text encoding contract for the SDK.
It does not own retrieval, storage, or engine lifecycle behavior.
"""

from .base import LateTextEncoder
from .callable import CallableLateTextEncoder
from .colbert import (
    DEFAULT_COLBERT_MODEL_NAME,
    ColBERTTextEncoder,
)
from .registry import available_encoder_kinds, open_encoder, register_encoder

__all__ = [
    "available_encoder_kinds",
    "CallableLateTextEncoder",
    "ColBERTTextEncoder",
    "DEFAULT_COLBERT_MODEL_NAME",
    "LateTextEncoder",
    "open_encoder",
    "register_encoder",
]
