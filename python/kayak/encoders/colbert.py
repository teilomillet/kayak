"""Owns the first-party ColBERT text encoder for the public Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import warnings

from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
import torch

from kayak_bridge.api_types import DocIdsInput, DocTextsInput, TokenMatrixInput
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


def _tensor_to_token_ids(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in tensor.detach().cpu().tolist())


@dataclass(frozen=True, slots=True)
class ColBERTTextEncoder:
    """Encode text with a ColBERT checkpoint into late-interaction objects.

    Parameters
    ----------
    model_name:
        Hugging Face repo id for the ColBERT checkpoint to load.
    checkpoint:
        Optional prebuilt ColBERT checkpoint object. When provided, Kayak uses
        it directly instead of loading ``model_name``.
    gpus:
        ColBERT runtime GPU count. The public SDK defaults to CPU-friendly
        single-process use with ``gpus=0``.
    """

    model_name: str = DEFAULT_COLBERT_MODEL_NAME
    checkpoint: Checkpoint | None = None
    gpus: int = 0

    def _effective_checkpoint(self) -> Checkpoint:
        if self.checkpoint is not None:
            return self.checkpoint
        return _cached_checkpoint(self.model_name, self.gpus)

    def encode_query(self, text: str) -> LateQuery:
        """Encode one query string into ``LateQuery`` using ColBERT."""
        checkpoint = self._effective_checkpoint()
        with torch.inference_mode():
            encoded = checkpoint.queryFromText([text], to_cpu=True)
        return query(_tensor_to_vectors(encoded[0]), text=text)

    def _encode_document_tensor_and_token_ids(
        self,
        text: str,
    ) -> tuple[torch.Tensor, tuple[int, ...] | None]:
        checkpoint = self._effective_checkpoint()
        doc_tokenizer = getattr(checkpoint, "doc_tokenizer", None)
        doc_encoder = getattr(checkpoint, "doc", None)
        if doc_tokenizer is None or doc_encoder is None:
            with torch.inference_mode():
                encoded = checkpoint.docFromText([text], to_cpu=True)
            return encoded[0], None

        with torch.inference_mode():
            input_ids, attention_mask = doc_tokenizer.tensorize([text])
            encoded = doc_encoder(
                input_ids,
                attention_mask,
                keep_dims=True,
                to_cpu=True,
            )
        return encoded[0], _tensor_to_token_ids(input_ids[0])

    def encode_document_vectors(self, text: str) -> TokenMatrixInput:
        """Encode one document string into token-level vectors using ColBERT."""
        encoded, _ = self._encode_document_tensor_and_token_ids(text)
        return _tensor_to_vectors(encoded)

    def encode_documents(
        self,
        doc_ids: DocIdsInput,
        texts: DocTextsInput,
    ) -> LateDocuments:
        """Encode aligned document ids and texts into ``LateDocuments``."""
        text_rows = tuple(str(text) for text in texts)
        token_vectors: list[list[list[float]]] = []
        token_id_rows: list[tuple[int, ...]] = []
        saw_missing_token_ids = False
        for text in text_rows:
            encoded, token_ids = self._encode_document_tensor_and_token_ids(text)
            token_vectors.append(_tensor_to_vectors(encoded))
            if token_ids is None:
                saw_missing_token_ids = True
            else:
                token_id_rows.append(token_ids)
        return documents(
            doc_ids,
            tuple(token_vectors),
            texts=text_rows,
            token_ids=None if saw_missing_token_ids else tuple(token_id_rows),
        )
