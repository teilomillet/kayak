"""Resolve artifacts and allocate the optional inference libraries on demand."""

from pathlib import Path
from typing import Protocol, cast

from .. import bundle
from ..errors import InputError, ModelLoadError
from ._model import Encoder, Model, Tokenizer


def load(
    model: str | Path = bundle.DEFAULT_MODEL,
    *,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 1,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Model:
    """Load the pinned release or a directory containing kayak.json and its heads.

    The full Qwen3 encoder runs in this process. Downloads occur only here.
    CUDA, MPS, and CPU are selectable; actual support requires hardware validation.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise InputError("batch_size must be a positive integer")
    if device not in {"auto", "cpu", "cuda", "mps"} or dtype not in {
        "auto",
        "float32",
        "float16",
        "bfloat16",
    }:
        raise InputError("unsupported device or dtype")
    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        from .._heads import load_heads
    except ImportError as exc:
        raise ModelLoadError("local loading requires: pip install 'kayak[local]'") from exc
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ModelLoadError("MPS is not available")
    if dtype == "auto":
        if device == "cpu":
            dtype = "float32"
        elif device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        else:
            dtype = "float16"
    spec, checkpoint = bundle.resolve(model, cache_dir=cache_dir, local_files_only=local_files_only)
    try:
        if Path(spec.encoder_id).exists():
            raise ModelLoadError("a local path shadows the pinned encoder Hub ID")
        # Read and validate the small artifacts before allocating the 8B encoder.
        options = dict(
            revision=spec.encoder_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        config = AutoConfig.from_pretrained(spec.encoder_id, **options)
        if config.model_type != "qwen3" or config.hidden_size != spec.hidden_size:
            raise ModelLoadError("encoder architecture/dimension does not match the manifest")
        if config.max_position_embeddings < spec.max_length:
            raise ModelLoadError("encoder context limit is smaller than the manifest limit")
        state_head, action_head, scale = load_heads(
            checkpoint, spec.hidden_size, device, spec.encoder_id
        )
        tokenizer = cast(_TokenizerFactory, AutoTokenizer.from_pretrained)(
            spec.encoder_id, **options
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            raise ModelLoadError("encoder tokenizer must define a padding token")
        encoder = (
            AutoModel.from_pretrained(
                spec.encoder_id,
                **options,
                config=config,
                torch_dtype=getattr(torch, dtype),
                use_safetensors=True,
                attn_implementation="sdpa",
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        return Model(
            spec=spec,
            # Transformers' Auto factories are dynamically typed. The checked
            # architecture and tokenizer settings establish these narrow interfaces.
            tokenizer=tokenizer,
            encoder=cast(Encoder, encoder),
            state_head=state_head,
            action_head=action_head,
            scale=scale,
            batch_size=batch_size,
        )
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"could not load CLM artifacts ({type(exc).__name__})") from exc


class _TokenizerFactory(Protocol):
    """The Transformers factory signature used only at this loading boundary."""

    def __call__(self, identifier: str, **options: object) -> Tokenizer: ...
