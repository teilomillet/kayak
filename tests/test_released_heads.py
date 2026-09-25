"""Optional real-weight checks against an independent, portable reference."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kayak.bundle import ModelSpec

if TYPE_CHECKING:
    from kayak._heads import ProjectionHead

pytestmark = pytest.mark.inference


@pytest.fixture
def released_checkpoint() -> tuple[Path, ModelSpec]:
    checkpoint = os.environ.get("KAYAK_TEST_HEADS")
    if checkpoint is None:
        pytest.skip("set KAYAK_TEST_HEADS to the released CLM_v0.1-8B.pt")
    pytest.importorskip("torch")
    spec = ModelSpec.model_validate_json(
        files("kayak").joinpath("models/clm-v0.1-8b.json").read_text()
    )
    return Path(checkpoint), spec


def test_released_heads_match_independent_reference(
    released_checkpoint: tuple[Path, ModelSpec],
) -> None:
    from scripts.validate_model import head_reference

    report = head_reference(*released_checkpoint)
    assert report["status"] == "passed"
    assert report["atol"] == 1e-5
    assert report["runtime_reference_dtype"] == "float32"
    assert report["recorded_reference_dtype"] == "float64"


@pytest.mark.parametrize("scale_factor", [1.01, float("nan")])
def test_head_reference_rejects_runtime_regressions(
    released_checkpoint: tuple[Path, ModelSpec],
    monkeypatch: pytest.MonkeyPatch,
    scale_factor: float,
) -> None:
    from kayak._heads import load_heads
    from scripts import validate_model

    def changed_heads(
        path: Path, hidden_size: int, device: str, encoder_id: str
    ) -> tuple[ProjectionHead, ProjectionHead, float]:
        sh, ah, scale = load_heads(path, hidden_size, device, encoder_id)
        return sh, ah, scale * scale_factor

    monkeypatch.setattr(validate_model, "load_heads", changed_heads)
    with pytest.raises(AssertionError, match="FP32 runtime head reference"):
        validate_model.head_reference(*released_checkpoint)


def test_head_reference_rejects_changed_checkpoint(
    released_checkpoint: tuple[Path, ModelSpec], tmp_path: Path
) -> None:
    from scripts.validate_model import head_reference

    _, spec = released_checkpoint
    changed = tmp_path / "heads.pt"
    changed.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="only to the selected release"):
        head_reference(changed, spec)
