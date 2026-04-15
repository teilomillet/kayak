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


def _trim_zero_padded_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError("trimmed ColBERT document tensors must be 2D")

    nonzero_rows = torch.any(tensor != 0, dim=1)
    if not bool(torch.any(nonzero_rows)):
        return tensor[:0]

    last_nonzero_row = int(torch.nonzero(nonzero_rows, as_tuple=False)[-1].item())
    return tensor[: last_nonzero_row + 1]


def encode_query_text(
    text: str, model_name: str = DEFAULT_MODEL_NAME
) -> list[list[float]]:
    checkpoint = get_checkpoint(model_name)

    with torch.inference_mode():
        encoded = checkpoint.queryFromText([text], to_cpu=True)

    return _tensor_to_vectors(encoded[0])


def encode_document_text(
    text: str, model_name: str = DEFAULT_MODEL_NAME
) -> list[list[float]]:
    checkpoint = get_checkpoint(model_name)

    with torch.inference_mode():
        encoded = checkpoint.docFromText([text], to_cpu=True)

    return _tensor_to_vectors(encoded[0])


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
