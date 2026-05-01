"""Shared public-facing type aliases for Python late-interaction inputs.

These aliases are intentionally broad enough to match the runtime conversion
code in the Python SDK while still giving editors and readers names that are
more informative than raw ``object`` annotations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np


ScalarInput: TypeAlias = int | float

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    TensorInput: TypeAlias = torch.Tensor
else:  # pragma: no cover - runtime alias only
    TensorInput: TypeAlias = object


TokenMatrixInput: TypeAlias = (
    np.ndarray | TensorInput | Sequence[Sequence[ScalarInput]]
)
TokenValuesInput: TypeAlias = np.ndarray | TensorInput | Sequence[ScalarInput]
TokenIdValuesInput: TypeAlias = np.ndarray | TensorInput | Sequence[int]
DocumentTokenIdsInput: TypeAlias = (
    np.ndarray | TensorInput | Sequence[Sequence[int]]
)
QueryBatchInput: TypeAlias = np.ndarray | TensorInput | Sequence[TokenMatrixInput]
DocumentMatricesInput: TypeAlias = (
    np.ndarray | TensorInput | Sequence[TokenMatrixInput]
)
DocIdsInput: TypeAlias = Sequence[object]
DocOffsetsInput: TypeAlias = Sequence[int] | np.ndarray
DocTextsInput: TypeAlias = Sequence[object]
QueryTextsInput: TypeAlias = Sequence[object]
MetadataRow: TypeAlias = Mapping[str, object]
MetadataRowsInput: TypeAlias = Sequence[MetadataRow | None] | None
MetadataFilterInput: TypeAlias = MetadataRow | None
