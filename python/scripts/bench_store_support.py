from __future__ import annotations

from pathlib import Path
import time

import numpy as np

import kayak


def dim_vector(vector_dim: int, hot_index: int) -> np.ndarray:
    vector = np.zeros(vector_dim, dtype=np.float32)
    vector[hot_index % vector_dim] = np.float32(1.0)
    return vector


def build_documents(
    *,
    document_count: int,
    vectors_per_document: int,
    vector_dim: int,
) -> tuple[kayak.LateDocuments, list[dict[str, object]]]:
    doc_ids = [f"doc-{index:05d}" for index in range(document_count)]
    matrices = []
    texts = []
    metadata = []
    for document_index in range(document_count):
        matrix = np.stack(
            [
                dim_vector(
                    vector_dim,
                    hot_index=document_index * vectors_per_document + vector_index,
                )
                for vector_index in range(vectors_per_document)
            ]
        )
        matrices.append(matrix)
        texts.append(f"document {document_index}")
        metadata.append({"bucket": document_index % 4, "group": "bench"})
    return kayak.documents(doc_ids, matrices, texts=texts), metadata


def build_query(
    *,
    vectors_per_document: int,
    vector_dim: int,
    target_document_index: int = 0,
) -> kayak.LateQuery:
    return kayak.query(
        np.stack(
            [
                dim_vector(
                    vector_dim,
                    hot_index=target_document_index * vectors_per_document + vector_index,
                )
                for vector_index in range(vectors_per_document)
            ]
        ),
        text=f"doc-{target_document_index:05d}",
    )


def timed_mean_seconds(
    fn,
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / float(repeats)


def open_store(
    kind: str,
    *,
    path: Path | None,
    table_name: str = "late_documents",
) -> object:
    if kind in {"directory", "kayak"}:
        if path is None:
            raise ValueError(f"{kind} store benchmark requires a path")
        return kayak.open_store(kind, path=path)
    if kind in {"lance", "lancedb"}:
        if path is None:
            raise ValueError("lancedb store benchmark requires a path")
        return kayak.open_store(kind, path=path, table_name=table_name)
    return kayak.open_store(kind)
