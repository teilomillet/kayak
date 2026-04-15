"""Compact JSON round-trip for token-level vector matrices.

This is only for compatibility adapters that must store exact late-interaction
matrices inside a single-vector-native system such as Chroma.
"""

from __future__ import annotations

import json

import numpy as np


def encode_token_matrix_json(matrix: np.ndarray) -> str:
    if matrix.ndim != 2:
        raise ValueError("token matrix must be 2D")

    payload = {
        "vector_count": int(matrix.shape[0]),
        "vector_dim": int(matrix.shape[1]),
        "values": np.asarray(matrix, dtype=np.float32).reshape(-1).tolist(),
    }
    return json.dumps(payload, separators=(",", ":"))


def decode_token_matrix_json(payload: object) -> np.ndarray:
    if not isinstance(payload, str):
        raise ValueError("stored token matrix payload must be a JSON string")

    decoded = json.loads(payload)
    vector_count = int(decoded["vector_count"])
    vector_dim = int(decoded["vector_dim"])
    values = np.asarray(decoded["values"], dtype=np.float32)
    matrix = values.reshape(vector_count, vector_dim)
    matrix.setflags(write=False)
    return matrix
