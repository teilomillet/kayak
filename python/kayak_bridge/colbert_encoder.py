from __future__ import annotations

from functools import lru_cache
import warnings

from .cache_paths import configure_local_caches

configure_local_caches()

from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
import torch


DEFAULT_MODEL_NAME = "colbert-ir/colbertv2.0"


def _make_cpu_config() -> ColBERTConfig:
    return ColBERTConfig(gpus=0)


@lru_cache(maxsize=4)
def get_checkpoint(model_name: str = DEFAULT_MODEL_NAME) -> Checkpoint:
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
    return Checkpoint(model_name, colbert_config=_make_cpu_config(), verbose=0)


def _tensor_to_vectors(tensor: torch.Tensor) -> list[list[float]]:
    return tensor.detach().cpu().tolist()


def _tensor_to_token_ids(tensor: torch.Tensor) -> list[int]:
    return [int(token_id) for token_id in tensor.detach().cpu().tolist()]


def _trim_zero_padded_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError("trimmed ColBERT document tensors must be 2D")

    nonzero_rows = torch.any(tensor != 0, dim=1)
    if not bool(torch.any(nonzero_rows)):
        return tensor[:0]

    last_nonzero_row = int(torch.nonzero(nonzero_rows, as_tuple=False)[-1].item())
    return tensor[: last_nonzero_row + 1]


def _trim_encoded_document_with_token_ids(
    encoded: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    trimmed = _trim_zero_padded_rows(encoded)
    return trimmed, _tensor_to_token_ids(token_ids[: int(trimmed.shape[0])])


def encode_query_text(
    text: str, model_name: str = DEFAULT_MODEL_NAME
) -> list[list[float]]:
    checkpoint = get_checkpoint(model_name)

    with torch.inference_mode():
        encoded = checkpoint.queryFromText([text], to_cpu=True)

    return _tensor_to_vectors(encoded[0])


def encode_query_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    batch_size: int = 32,
) -> tuple[list[list[float]], ...]:
    if batch_size <= 0:
        raise ValueError("query batch_size must be positive")
    if not texts:
        return ()

    checkpoint = get_checkpoint(model_name)
    encoded_queries: list[list[list[float]]] = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded_batch = checkpoint.queryFromText(batch_texts, to_cpu=True)
            for encoded in encoded_batch:
                encoded_queries.append(_tensor_to_vectors(encoded))

    return tuple(encoded_queries)


def encode_query_texts_as_tensors(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    batch_size: int = 32,
) -> tuple[torch.Tensor, ...]:
    if batch_size <= 0:
        raise ValueError("query batch_size must be positive")
    if not texts:
        return ()

    checkpoint = get_checkpoint(model_name)
    encoded_queries: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded_batch = checkpoint.queryFromText(batch_texts, to_cpu=True)
            for encoded in encoded_batch:
                encoded_queries.append(encoded.detach().cpu().contiguous())

    return tuple(encoded_queries)


def encode_document_text(
    text: str, model_name: str = DEFAULT_MODEL_NAME
) -> list[list[float]]:
    checkpoint = get_checkpoint(model_name)

    with torch.inference_mode():
        encoded = checkpoint.docFromText([text], to_cpu=True)

    return _tensor_to_vectors(encoded[0])


def encode_document_text_with_token_ids(
    text: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[list[list[float]], list[int]]:
    checkpoint = get_checkpoint(model_name)
    doc_tokenizer = getattr(checkpoint, "doc_tokenizer", None)
    doc_encoder = getattr(checkpoint, "doc", None)
    if doc_tokenizer is None or doc_encoder is None:
        raise RuntimeError("ColBERT checkpoint does not expose document token ids")

    with torch.inference_mode():
        input_ids, attention_mask = doc_tokenizer.tensorize([text])
        encoded_batch = doc_encoder(
            input_ids,
            attention_mask,
            keep_dims=True,
            to_cpu=True,
        )

    encoded, token_ids = _trim_encoded_document_with_token_ids(
        encoded_batch[0],
        input_ids[0],
    )
    return _tensor_to_vectors(encoded), token_ids


def encode_document_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    batch_size: int = 8,
) -> tuple[list[list[float]], ...]:
    if batch_size <= 0:
        raise ValueError("document batch_size must be positive")
    if not texts:
        return ()

    checkpoint = get_checkpoint(model_name)
    encoded_documents: list[list[list[float]]] = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded_batch = checkpoint.docFromText(batch_texts, to_cpu=True)
            for encoded in encoded_batch:
                encoded_documents.append(
                    _tensor_to_vectors(_trim_zero_padded_rows(encoded))
                )

    return tuple(encoded_documents)


def encode_document_texts_with_token_ids(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    batch_size: int = 8,
) -> tuple[tuple[list[list[float]], list[int]], ...]:
    if batch_size <= 0:
        raise ValueError("document batch_size must be positive")
    if not texts:
        return ()

    checkpoint = get_checkpoint(model_name)
    doc_tokenizer = getattr(checkpoint, "doc_tokenizer", None)
    doc_encoder = getattr(checkpoint, "doc", None)
    if doc_tokenizer is None or doc_encoder is None:
        raise RuntimeError("ColBERT checkpoint does not expose document token ids")

    encoded_documents: list[tuple[list[list[float]], list[int]]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            input_ids, attention_mask = doc_tokenizer.tensorize(batch_texts)
            encoded_batch = doc_encoder(
                input_ids,
                attention_mask,
                keep_dims=True,
                to_cpu=True,
            )
            for encoded, row_token_ids in zip(
                encoded_batch,
                input_ids,
                strict=True,
            ):
                trimmed, token_ids = _trim_encoded_document_with_token_ids(
                    encoded,
                    row_token_ids,
                )
                encoded_documents.append((_tensor_to_vectors(trimmed), token_ids))

    return tuple(encoded_documents)


def encode_document_texts_with_token_id_tensors(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    batch_size: int = 8,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if batch_size <= 0:
        raise ValueError("document batch_size must be positive")
    if not texts:
        return ()

    checkpoint = get_checkpoint(model_name)
    doc_tokenizer = getattr(checkpoint, "doc_tokenizer", None)
    doc_encoder = getattr(checkpoint, "doc", None)
    if doc_tokenizer is None or doc_encoder is None:
        raise RuntimeError("ColBERT checkpoint does not expose document token ids")

    encoded_documents: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            input_ids, attention_mask = doc_tokenizer.tensorize(batch_texts)
            encoded_batch = doc_encoder(
                input_ids,
                attention_mask,
                keep_dims=True,
                to_cpu=True,
            )
            for encoded, row_token_ids in zip(
                encoded_batch,
                input_ids,
                strict=True,
            ):
                trimmed = _trim_zero_padded_rows(encoded)
                token_ids = row_token_ids[: int(trimmed.shape[0])]
                encoded_documents.append(
                    (
                        trimmed.detach().cpu().contiguous(),
                        token_ids.detach().cpu().contiguous(),
                    )
                )

    return tuple(encoded_documents)
