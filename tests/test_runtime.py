from __future__ import annotations

import json
from pathlib import Path

import pytest

from kayak import Choice, InputError, Model, ModelClosedError, ModelLoadError, load
from kayak.bundle import resolve

pytestmark = pytest.mark.inference


def test_real_tiny_qwen_local_decision(tiny_model: Model, questions: dict[str, Choice]) -> None:
    result = tiny_model.decide(state="charged twice", questions=questions)
    assert set(result.answers["route"].scores) == {"billing", "support"}
    assert result.calibration == "none"
    assert result.model.revision == "fixture-v1"
    assert result.input_tokens > 0


def test_real_tiny_qwen_mps_and_live_http(tiny_bundle: Path, questions: dict[str, Choice]) -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    from scripts.validate_model import memory_snapshot, validate_http

    question = questions["route"]
    case = ("charged twice", question.instructions, question.criteria)
    with load(tiny_bundle, device="mps") as model:
        direct = model.decide(state=case[0], questions={"decision": question})
        assert torch.device(direct.model.device).type == "mps"
        assert direct.model.dtype == "float16"
        assert direct.input_tokens > 0
        assert set(direct.answers["decision"].scores) == set(question.criteria)
        memory = memory_snapshot(direct.model.device)
        for counter in ("mps_current_tensor_bytes", "mps_current_driver_bytes"):
            value = memory[counter]
            assert value is not None and value > 0
        repeated = model.decide(state=case[0], questions={"decision": question})
        assert repeated.answers == direct.answers
        assert validate_http(model, case, direct)


def test_padding_and_batching_preserve_scores(
    tiny_bundle: Path, questions: dict[str, Choice]
) -> None:
    with load(tiny_bundle, device="cpu", batch_size=1) as single:
        expected = single.decide(state="charged twice", questions=questions)
    with load(tiny_bundle, device="cpu", batch_size=4) as batched:
        actual = batched.decide(state="charged twice", questions=questions)
    assert actual.answers["route"].scores == pytest.approx(
        expected.answers["route"].scores, abs=1e-6
    )


def test_candidate_order_and_question_ids(tiny_model: Model, questions: dict[str, Choice]) -> None:
    original = tiny_model.decide(state="charged twice", questions=questions)
    q = questions["route"]
    swapped = Choice(instructions=q.instructions, criteria=dict(reversed(list(q.criteria.items()))))
    changed = tiny_model.decide(state="charged twice", questions={"renamed": swapped})
    assert changed.answers["renamed"].scores == pytest.approx(
        original.answers["route"].scores, abs=1e-6
    )


def test_overflow_is_rejected_before_forward(
    tiny_model: Model, questions: dict[str, Choice], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiny_model._encoder, "forward", lambda **kw: pytest.fail("ran inference"))
    with pytest.raises(InputError, match="never silently truncated"):
        tiny_model.decide(state="charged " * 2048, questions=questions)
    with pytest.raises(InputError, match="16384"):
        tiny_model.decide(
            state="charged " * 1024, questions={str(i): questions["route"] for i in range(17)}
        )


def test_close_is_idempotent(tiny_model: Model, questions: dict[str, Choice]) -> None:
    tiny_model.close()
    tiny_model.close()
    with pytest.raises(ModelClosedError):
        tiny_model.decide(state="charged twice", questions=questions)


def test_checkpoint_corruption_is_rejected(tiny_bundle: Path) -> None:
    (tiny_bundle / "heads.pt").write_bytes(b"corrupt")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        load(tiny_bundle)


def test_wrong_encoder_dimension_rejected(tiny_bundle: Path) -> None:
    path = tiny_bundle / "kayak.json"
    manifest = json.loads(path.read_text())
    manifest["hidden_size"] = 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ModelLoadError, match="dimension"):
        load(tiny_bundle)


def test_unknown_bundle_fails_without_fallback(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        resolve(tmp_path, local_files_only=True)
