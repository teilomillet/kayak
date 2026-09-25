"""Resolve the supported release or an explicit local CLM bundle."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from .decisions import Contract
from .errors import ModelLoadError

DEFAULT_MODEL = "Contrastive-LM/CLM-v0.1-8B"


class ModelSpec(Contract):
    format_version: Literal[1] = 1
    family: Literal["clm-qwen3"] = "clm-qwen3"
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    checkpoint: str
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    encoder_id: str = Field(pattern=r"^[\w-]+/[\w.-]+$")
    encoder_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    hidden_size: int = Field(gt=0)
    max_length: int = Field(gt=0, le=2048)
    input_recipe: Literal["clm-choice-v1"] = "clm-choice-v1"

    @field_validator("checkpoint")
    @classmethod
    def filename(cls, value: str) -> str:
        if not value or Path(value).name != value or "\\" in value or value in {".", ".."}:
            raise ValueError("checkpoint must be a filename within the bundle directory")
        return value

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(
    model: str | Path, *, cache_dir: str | Path | None = None, local_files_only: bool = False
) -> tuple[ModelSpec, Path]:
    """Resolve and integrity-check heads before allocating the encoder."""
    try:
        if str(model) == DEFAULT_MODEL:
            spec = ModelSpec.model_validate_json(
                files("kayak").joinpath("models/clm-v0.1-8b.json").read_text()
            )
            from huggingface_hub import hf_hub_download

            checkpoint = Path(
                hf_hub_download(
                    repo_id=spec.model_id,
                    filename=spec.checkpoint,
                    revision=spec.revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        else:
            directory = Path(model)
            spec = ModelSpec.model_validate_json((directory / "kayak.json").read_text())
            checkpoint = directory / spec.checkpoint
        if sha256(checkpoint) != spec.checkpoint_sha256:
            raise ModelLoadError("checkpoint SHA-256 does not match the model manifest")
        return spec, checkpoint
    except ModelLoadError:
        raise
    except ImportError as exc:
        raise ModelLoadError("local loading requires: pip install 'kayak[local]'") from exc
    except (OSError, ValidationError) as exc:
        raise ModelLoadError(
            f"cannot resolve model bundle ({type(exc).__name__}): {model}"
        ) from exc
