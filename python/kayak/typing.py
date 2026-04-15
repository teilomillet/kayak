"""Stable public type aliases for Kayak Python SDK integration code.

Import from ``kayak.typing`` when annotating:
- custom query and document encoder wrappers
- helper functions that build ``LateQuery`` or ``LateDocuments``
- application code that passes metadata filters or aligned text/id rows

Common aliases include ``TokenMatrixInput``, ``DocumentMatricesInput``,
``DocIdsInput``, ``DocTextsInput``, and ``MetadataRowsInput``.

This module owns only stable user-facing aliases for the Python SDK. It does
not expose backend-specific client protocols or internal bridge payload types.
"""

from __future__ import annotations

from kayak_bridge.api_types import (
    DocIdsInput,
    DocOffsetsInput,
    DocTextsInput,
    DocumentMatricesInput,
    MetadataFilterInput,
    MetadataRow,
    MetadataRowsInput,
    QueryBatchInput,
    QueryTextsInput,
    ScalarInput,
    TensorInput,
    TokenMatrixInput,
    TokenValuesInput,
)

_ALIAS_SUMMARIES: dict[str, str] = {
    "ScalarInput": "One scalar numeric value accepted by the public conversion helpers.",
    "TensorInput": "Tensor-like input accepted from optional tensor libraries such as PyTorch.",
    "TokenMatrixInput": "One query or document token matrix before Kayak normalizes it into a 2D float32 array.",
    "TokenValuesInput": "One flat token-value buffer such as the values used by `flat_dim128` queries.",
    "QueryBatchInput": "One batch of queries represented as matrices or tensor-like inputs.",
    "DocumentMatricesInput": "Aligned document token matrices before they are packed into a searchable index.",
    "DocIdsInput": "Aligned document ids accepted by document, store, and retriever APIs.",
    "DocOffsetsInput": "Packed document boundaries used when constructing one packed late-interaction index directly.",
    "DocTextsInput": "Aligned document texts accepted by document, encoder, store, and retriever APIs.",
    "QueryTextsInput": "Aligned query texts accepted by text-batch retriever APIs.",
    "MetadataRow": "One metadata mapping attached to one document row.",
    "MetadataRowsInput": "Aligned optional metadata rows attached during upsert or ingest.",
    "MetadataFilterInput": "One simple metadata filter mapping used to load or search one store subset.",
}

_ALIAS_TYPE_TEXT: dict[str, str] = {
    "ScalarInput": "int | float",
    "TensorInput": "torch.Tensor | tensor-like input",
    "TokenMatrixInput": "numpy.ndarray | torch.Tensor | Sequence[Sequence[int | float]]",
    "TokenValuesInput": "numpy.ndarray | torch.Tensor | Sequence[int | float]",
    "QueryBatchInput": "numpy.ndarray | torch.Tensor | Sequence[TokenMatrixInput]",
    "DocumentMatricesInput": "numpy.ndarray | torch.Tensor | Sequence[TokenMatrixInput]",
    "DocIdsInput": "Sequence[object]",
    "DocOffsetsInput": "Sequence[int] | numpy.ndarray",
    "DocTextsInput": "Sequence[object]",
    "QueryTextsInput": "Sequence[object]",
    "MetadataRow": "Mapping[str, object]",
    "MetadataRowsInput": "Sequence[MetadataRow | None] | None",
    "MetadataFilterInput": "MetadataRow | None",
}

__all__ = [
    "DocIdsInput",
    "DocOffsetsInput",
    "DocTextsInput",
    "DocumentMatricesInput",
    "MetadataFilterInput",
    "MetadataRow",
    "MetadataRowsInput",
    "QueryBatchInput",
    "QueryTextsInput",
    "ScalarInput",
    "TensorInput",
    "TokenMatrixInput",
    "TokenValuesInput",
]
