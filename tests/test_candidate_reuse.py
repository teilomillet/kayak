from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from kayak import Choice, InferenceError, InputError, ModelClosedError, load
from kayak.runtime._model import Encoder, EncoderOutput

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice
    from torch import dtype as TorchDtype

pytestmark = pytest.mark.inference


class RecordingEncoder:
    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.batch_rows: list[int] = []
        self.fail_next = False

    @property
    def device(self) -> TorchDevice:
        return self.encoder.device

    @property
    def dtype(self) -> TorchDtype:
        return self.encoder.dtype

    def __call__(self, *, use_cache: bool, **inputs: Tensor) -> EncoderOutput:
        self.batch_rows.append(inputs["input_ids"].shape[0])
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected encoder failure")
        return self.encoder(use_cache=use_cache, **inputs)


def test_reused_candidates_match_fresh_models(
    tiny_bundle: Path, questions: dict[str, Choice]
) -> None:
    question = questions["route"]
    renamed = Choice(
        instructions="Select team?",
        criteria=dict(zip(("first", "second"), question.criteria.values(), strict=True)),
    )
    reordered = Choice(
        instructions=question.instructions,
        criteria=dict(reversed(list(question.criteria.items()))),
    )
    changed = Choice(
        instructions=question.instructions,
        criteria={"billing": "billing refunds", "support": "technical outages"},
    )
    same_tokens = Choice(
        instructions=question.instructions,
        criteria={"billing": "billing  refunds", "support": "technical service outages"},
    )
    scenarios = [
        ("charged twice", questions, 3),
        ("technical service outages", questions, 1),
        ("charged twice", {"renamed": renamed}, 1),
        ("charged twice", {"route": reordered}, 3),
        ("charged twice", {"route": changed}, 3),
        ("charged twice", questions, 3),
        ("charged twice", {"route": same_tokens}, 1),
        ("charged twice", {"one": question, "two": changed}, 6),
        ("technical service", {"three": question, "four": changed}, 2),
        ("charged twice", questions, 3),
    ]
    with load(tiny_bundle, device="cpu", batch_size=1) as model:
        assert model._encoder is not None
        recorder = RecordingEncoder(model._encoder)
        model._encoder = recorder
        for state, current, expected_forwards in scenarios:
            start = len(recorder.batch_rows)
            actual = model.decide(state=state, questions=current)
            # A new resident model has no saved embeddings: the full computation
            # is an independent numerical reference for each history transition.
            with load(tiny_bundle, device="cpu", batch_size=1) as fresh:
                expected = fresh.decide(state=state, questions=current)
            assert actual == expected
            assert len(recorder.batch_rows) - start == expected_forwards
        assert set(recorder.batch_rows) == {1}
        model.close()
        assert model._candidate_cache is None
        with pytest.raises(ModelClosedError):
            model.decide(state="charged twice", questions=questions)


def test_reuse_preserves_question_boundaries(tiny_bundle: Path) -> None:
    descriptions = ["billing", "refunds", "technical", "service", "outages", "charged"]

    def questions(cut: int) -> dict[str, Choice]:
        return {
            "one": Choice(instructions="Which team?", criteria=dict(enumerated[:cut])),
            "two": Choice(instructions="Select team?", criteria=dict(enumerated[cut:])),
        }

    enumerated = [(str(i), text) for i, text in enumerate(descriptions)]
    with load(tiny_bundle, device="cpu", batch_size=1) as model:
        assert model._encoder is not None
        recorder = RecordingEncoder(model._encoder)
        model._encoder = recorder
        model.decide(state="charged twice", questions=questions(2))
        actual = model.decide(state="charged twice", questions=questions(3))
        assert len(recorder.batch_rows) == 8 + 2
        with load(tiny_bundle, device="cpu", batch_size=1) as fresh:
            assert actual == fresh.decide(state="charged twice", questions=questions(3))


def test_cached_calls_still_validate_and_recover_from_failure(
    tiny_bundle: Path, questions: dict[str, Choice]
) -> None:
    with load(tiny_bundle, device="cpu", batch_size=1) as model:
        assert model._encoder is not None
        recorder = RecordingEncoder(model._encoder)
        model._encoder = recorder
        expected = model.decide(state="charged twice", questions=questions)
        with pytest.raises(InputError, match="never silently truncated"):
            model.decide(state="charged " * 2048, questions=questions)
        assert len(recorder.batch_rows) == 3
        recorder.fail_next = True
        with pytest.raises(InferenceError, match="RuntimeError"):
            model.decide(state="charged twice", questions=questions)
        assert model.decide(state="charged twice", questions=questions) == expected

        other = {"route": Choice(instructions="Which team?", criteria={"a": "a", "b": "b"})}
        recorder.fail_next = True
        with pytest.raises(InferenceError, match="RuntimeError"):
            model.decide(state="charged twice", questions=other)
        actual = model.decide(state="charged twice", questions=other)
        with load(tiny_bundle, device="cpu", batch_size=1) as fresh:
            assert actual == fresh.decide(state="charged twice", questions=other)


@pytest.mark.parametrize("batch_size, expected_rows", [(2, [2, 1, 2, 1]), (4, [3, 3])])
def test_batched_encoder_layout_is_unchanged(
    tiny_bundle: Path,
    questions: dict[str, Choice],
    batch_size: int,
    expected_rows: list[int],
) -> None:
    with load(tiny_bundle, device="cpu", batch_size=batch_size) as model:
        assert model._encoder is not None
        recorder = RecordingEncoder(model._encoder)
        model._encoder = recorder
        first = model.decide(state="charged twice", questions=questions)
        assert model.decide(state="charged twice", questions=questions) == first
        assert recorder.batch_rows == expected_rows
        assert model._candidate_cache is None
