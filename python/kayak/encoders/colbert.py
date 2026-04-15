"""Owns the first-party ColBERT text encoder for the public Python SDK."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import warnings

from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
import torch

from kayak_bridge import LateDocuments, LateQuery, documents, query


DEFAULT_COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"


def _configure_colbert_runtime() -> None:
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy("file_system")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore", message="CUDA is not available.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="resource_tracker: There appear to be .* leaked semaphore objects.*",
        category=UserWarning,
    )


@lru_cache(maxsize=4)
def _cached_checkpoint(model_name: str, gpus: int) -> Checkpoint:
    _configure_colbert_runtime()
    return Checkpoint(
        model_name,
        colbert_config=ColBERTConfig(gpus=gpus),
        verbose=0,
    )


def _tensor_to_vectors(tensor: torch.Tensor) -> list[list[float]]:
    return tensor.detach().cpu().tolist()


@dataclass(frozen=True, slots=True)
class ColBERTTextEncoder:
    """Encodes text with a ColBERT checkpoint into late-interaction objects."""

    model_name: str = DEFAULT_COLBERT_MODEL_NAME
    checkpoint: object | None = None
    gpus: int = 0

    def _effective_checkpoint(self) -> object:
        if self.checkpoint is not None:
            return self.checkpoint
        return _cached_checkpoint(self.model_name, self.gpus)

    def encode_query(self, text: str) -> LateQuery:
        checkpoint = self._effective_checkpoint()
        with torch.inference_mode():
            encoded = checkpoint.queryFromText([text], to_cpu=True)
        return query(_tensor_to_vectors(encoded[0]), text=text)

    def encode_document_vectors(self, text: str) -> object:
        checkpoint = self._effective_checkpoint()
        with torch.inference_mode():
            encoded = checkpoint.docFromText([text], to_cpu=True)
        return _tensor_to_vectors(encoded[0])

    def encode_documents(
        self,
        doc_ids: Sequence[object],
        texts: Sequence[object],
    ) -> LateDocuments:
        text_rows = tuple(str(text) for text in texts)
        token_vectors = tuple(
            self.encode_document_vectors(text) for text in text_rows
        )
        return documents(
            doc_ids,
            token_vectors,
            texts=text_rows,
        )
