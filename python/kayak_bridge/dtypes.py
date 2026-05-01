from __future__ import annotations

import numpy as np


# Mirrors the current Mojo scalar choices from kayak/numeric/scalars.mojo.
VECTOR_DTYPE = np.float32
SCORE_DTYPE = np.float32
INDEX_OFFSET_DTYPE = np.int64
TOKEN_ID_DTYPE = np.int64
VECTOR_SCALAR_NAME = "Float32"
MIN_SCORE = np.float32(np.finfo(np.float32).min)
FLAT_DIM128_VECTOR_DIM = 128
