from __future__ import annotations

import json
from importlib.resources import files

import pytest
from pydantic import ValidationError

from kayak.bundle import ModelSpec


def test_published_model_identity_is_pinned() -> None:
    spec = ModelSpec.model_validate_json(
        files("kayak").joinpath("models/clm-v0.1-8b.json").read_text()
    )
    assert spec.model_id == "Contrastive-LM/CLM-v0.1-8B"
    assert len(spec.revision) == len(spec.encoder_revision) == 40
    assert (
        spec.checkpoint_sha256 == "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"format_version": 2},
        {"family": "unknown"},
        {"input_recipe": "unknown"},
        {"checkpoint": "../heads.pt"},
        {"checkpoint_sha256": "bad"},
        {"encoder_revision": "main"},
        {"extra": True},
    ],
)
def test_manifest_rejects_unsupported_or_unpinned_inputs(change: dict[str, object]) -> None:
    spec = json.loads(files("kayak").joinpath("models/clm-v0.1-8b.json").read_text())
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({**spec, **change})
