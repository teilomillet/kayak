"""The experiment must change only the prepared state/question order."""

from collections.abc import Mapping

import pytest

from benchmarks.question_first import question_first, question_first_request
from kayak import Choice, DecisionRequest, DecisionResult, InputError
from kayak.decisions import request_from
from kayak.runtime._preparation import prepare_texts


def test_question_first_preserves_exact_text_candidates_and_source() -> None:
    request = DecisionRequest(
        state="  Ma carte ne fonctionne pas.\nEncore.  ",
        questions={
            "intent": Choice(
                instructions="  Quel problème ?\nChoisissez.  ",
                criteria={"z": "  Retrait refusé  ", "a": "Paiement refusé"},
            )
        },
    )
    original = request.model_dump()
    transformed = question_first_request(request)
    assert prepare_texts(transformed) == [
        "Quel problème ?\nChoisissez.\n\nMa carte ne fonctionne pas.\nEncore.",
        "  Retrait refusé  ",
        "Paiement refusé",
    ]
    assert list(transformed.questions["intent"].criteria) == ["z", "a"]
    assert request.model_dump() == original


def test_multiple_questions_cannot_silently_change_experiment_semantics() -> None:
    question = Choice(instructions="Which intent?", criteria={"a": "Alpha"})
    with pytest.raises(InputError, match="exactly one question"):
        question_first_request(
            DecisionRequest(state="Customer", questions={"one": question, "two": question})
        )


def test_transformed_call_is_restored_even_when_backend_fails() -> None:
    class Recorder:
        request: DecisionRequest | None = None

        def decide(
            self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
        ) -> DecisionResult:
            self.request = request_from(state, questions)
            raise RuntimeError("controlled failure")

    model = Recorder()
    original = model.decide
    with pytest.raises(RuntimeError, match="controlled failure"), question_first(model):
        model.decide(
            state="Customer text",
            questions={"intent": Choice(instructions="Which intent?", criteria={"a": "Alpha"})},
        )
    assert model.decide == original
    assert model.request is not None
    assert model.request.state == "Which intent?"
    assert model.request.questions["intent"].instructions == "Customer text"
