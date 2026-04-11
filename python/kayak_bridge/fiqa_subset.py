from __future__ import annotations

from .beir_subset import build_beir_colbert_subset
from .colbert_encoder import DEFAULT_MODEL_NAME


DEFAULT_DATASET_ID = "beir/fiqa/test"


def build_fiqa_colbert_subset(
    query_limit: int = 6,
    negative_doc_limit: int = 64,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict:
    return build_beir_colbert_subset(
        dataset_id=dataset_id,
        slice_name="fiqa_real_subset",
        why="Real BEIR/FIQA subset encoded with ColBERTv2 on CPU for a public end-to-end smoke path in financial QA.",
        query_limit=query_limit,
        negative_doc_limit=negative_doc_limit,
        model_name=model_name,
    )
