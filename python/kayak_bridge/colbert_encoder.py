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
