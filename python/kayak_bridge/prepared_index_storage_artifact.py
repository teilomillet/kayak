"""Owns temporary flat dim128 artifacts for fast Mojo preparation.

This module exists only for the Python exact bridge cold path. The prepared
exact binding only needs document ids, document offsets, and flat dim128 token
values, so this artifact is intentionally narrower than the general packed
index store.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from .cache_paths import PYTHON_MOJO_CACHE, configure_local_caches
from .dtypes import FLAT_DIM128_VECTOR_DIM, INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from .layouts import INDEX_LAYOUT_PACKED

import numpy as np


PREPARED_PACKED_INDEX_ARTIFACT_ROOT = (
    PYTHON_MOJO_CACHE / "prepared_packed_index_artifacts"
)


def prepared_packed_index_artifact_root(*, module: object, index: "LateIndex") -> Path:
    configure_local_caches()
    return PREPARED_PACKED_INDEX_ARTIFACT_ROOT / (
        f"module_{id(module):x}_index_{id(index):x}"
    )


def write_prepared_packed_index_artifact(index: "LateIndex", *, root: Path) -> Path:
    if index.layout != INDEX_LAYOUT_PACKED:
        raise ValueError("prepared packed-index artifacts require packed layout")
    if index.vector_dim != FLAT_DIM128_VECTOR_DIM:
        raise ValueError("prepared packed-index artifacts require vector_dim=128")

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    _write_doc_ids(index, root=root)
    _write_doc_offsets(index, root=root)
    _write_token_values(index, root=root)
    return root


def remove_prepared_packed_index_artifact(root: Path | None) -> None:
    if root is None:
        return
    shutil.rmtree(root, ignore_errors=True)


def _write_doc_ids(index: "LateIndex", *, root: Path) -> None:
    payload = "".join(f"{doc_id}\n" for doc_id in index.doc_ids)
    (root / "doc_ids.tsv").write_text(payload, encoding="utf-8")


def _write_doc_offsets(index: "LateIndex", *, root: Path) -> None:
    little_endian_dtype = np.dtype(INDEX_OFFSET_DTYPE).newbyteorder("<")
    encoded = np.asarray(index.doc_offsets, dtype=little_endian_dtype, order="C")
    with (root / "doc_offsets.bin").open("wb") as handle:
        encoded.tofile(handle)


def _write_token_values(index: "LateIndex", *, root: Path) -> None:
    values = index.as_flat_token_values()
    little_endian_dtype = np.dtype(VECTOR_DTYPE).newbyteorder("<")
    encoded = np.asarray(values, dtype=little_endian_dtype, order="C")
    with (root / "token_values.bin").open("wb") as handle:
        encoded.tofile(handle)
