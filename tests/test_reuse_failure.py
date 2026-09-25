from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kayak import Choice, InferenceError, load
from kayak.runtime._model import Encoder, EncoderOutput

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice
    from torch import dtype as TorchDtype

pytestmark = pytest.mark.inference


@dataclass
class Output:
    last_hidden_state: Tensor


@dataclass
class OnceNonfiniteEncoder:
    encoder: Encoder
    calls: int = 0

    @property
    def device(self) -> TorchDevice:
        return self.encoder.device

    @property
    def dtype(self) -> TorchDtype:
        return self.encoder.dtype

    def __call__(self, *, use_cache: bool, **inputs: Tensor) -> EncoderOutput:
        output = self.encoder(use_cache=use_cache, **inputs)
        self.calls += 1
        if self.calls == 2:
            # Fail one candidate vector, after the state was encoded successfully.
            hidden = output.last_hidden_state.clone()
            hidden.fill_(float("nan"))
            return Output(hidden)
        return output


def test_rejected_candidate_output_does_not_poison_later_calls(
    tiny_bundle: Path, questions: dict[str, Choice]
) -> None:
    with load(tiny_bundle, device="cpu", batch_size=1) as model:
        assert model._encoder is not None
        model._encoder = OnceNonfiniteEncoder(model._encoder)
        with pytest.raises(InferenceError, match="finite"):
            model.decide(state="charged twice", questions=questions)
        actual = model.decide(state="charged twice", questions=questions)
        with load(tiny_bundle, device="cpu", batch_size=1) as fresh:
            assert actual == fresh.decide(state="charged twice", questions=questions)
