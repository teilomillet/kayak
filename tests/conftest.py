from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

import pytest
from hypothesis import settings

from kayak import Choice, Model

settings.register_profile("ci", max_examples=300, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


class PretrainedFactory(Protocol):
    def __call__(self, identifier: str | Path, **kwargs: object) -> object: ...


@pytest.fixture
def questions() -> dict[str, Choice]:
    return {
        "route": Choice(
            instructions="Which team?",
            criteria={"billing": "billing refunds", "support": "technical service outages"},
        )
    }


@pytest.fixture
def tiny_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Exercise the actual loader and a small Qwen3; replace only Hub resolution."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from kayak._heads import HeadOptions, ProjectionHead
    from kayak.bundle import sha256

    torch.set_num_threads(1)
    torch.manual_seed(7)
    encoder_dir = tmp_path / "encoder"
    config = transformers.Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=2048,
        pad_token_id=0,
    )
    transformers.Qwen3Model(config).save_pretrained(encoder_dir)
    tokens = [
        "[PAD]",
        "[UNK]",
        "Which",
        "team",
        "billing",
        "refunds",
        "technical",
        "service",
        "outages",
        "charged",
        "twice",
        "Ready",
        ".",
        "?",
        "Select",
        "readiness",
    ]
    tokenizer = Tokenizer(WordLevel({s: i for i, s in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
    ).save_pretrained(encoder_dir)
    options: HeadOptions = dict(
        hidden=16,
        width=24,
        depth=3,
        projection_dim=8,
        activation="gelu",
        layernorm=True,
        residual=False,
    )
    cfg = {**options, "hidden_size": 16, "model": "tests/tiny-qwen"}
    checkpoint = tmp_path / "heads.pt"
    torch.save(
        {
            "state_head": ProjectionHead(**options).state_dict(),
            "action_head": ProjectionHead(**options).state_dict(),
            "cfg": cfg,
            "logit_scale": torch.tensor(2.0),
        },
        checkpoint,
    )
    manifest = dict(
        format_version=1,
        family="clm-qwen3",
        model_id="tests/tiny-clm",
        revision="fixture-v1",
        checkpoint="heads.pt",
        checkpoint_sha256=sha256(checkpoint),
        encoder_id="tests/tiny-qwen",
        encoder_revision="a" * 40,
        hidden_size=16,
        max_length=2048,
        input_recipe="clm-choice-v1",
    )
    (tmp_path / "kayak.json").write_text(json.dumps(manifest))

    for cls in (transformers.AutoConfig, transformers.AutoTokenizer, transformers.AutoModel):
        original = cast(PretrainedFactory, cls.from_pretrained)

        def from_local(
            identifier: str | Path, _original: PretrainedFactory = original, **kwargs: object
        ) -> object:
            assert identifier == "tests/tiny-qwen"
            assert kwargs.pop("revision") == "a" * 40
            assert kwargs["trust_remote_code"] is False
            return _original(encoder_dir, **kwargs)

        monkeypatch.setattr(cls, "from_pretrained", from_local)
    return tmp_path


@pytest.fixture
def tiny_model(tiny_bundle: Path) -> Iterator[Model]:
    from kayak import load

    with load(tiny_bundle, device="cpu", batch_size=4) as model:
        yield model
