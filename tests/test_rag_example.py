"""Check an executed diagnostic intervention and its retained evidence."""

import json
import subprocess
import sys

from examples.evaluate_rag import answer, record_runs
from kayak.eval import RAGJudgments, assess_rag


def test_context_intervention_reruns_answer_without_rewriting_original() -> None:
    original, replay = record_runs()
    assert original.context is not None and replay.context is not None
    assert original.context.ids == ["profile"]
    assert replay.context.ids == ["profile", "invoice"]
    assert original.retrieval == replay.retrieval
    assert original.reranking == replay.reranking
    assert original.answer != replay.answer
    assert replay.answer == answer(replay.context)
    assert replay.replay is not None
    assert replay.replay.original_trace_sha256 == original.sha256
    assert replay.replay.changed_boundary == "context"
    labels = RAGJudgments(
        relevance={"profile": 0, "invoice": 1},
        evidence_sets=[["invoice"]],
        evidence_texts={"invoice": "Settings > Billing > Invoices"},
    )
    before, after = assess_rag(original, labels, k=1), assess_rag(replay, labels, k=1)
    assert before.source_coverage == {
        "retrieval": True,
        "reranking_top_k": False,
        "context": False,
    }
    assert before.required_text_retained is False
    assert after.source_coverage["context"] is True
    assert after.required_text_retained is True
    assert after.ordinary_aggregate_eligible is False
    assert after.answer_correct is None  # Execution and review are separate.


def test_example_runs_without_optional_inference_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from examples.evaluate_rag import main; main(); import sys; "
            "assert not {'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(rows) == 2
    before, after = (row["assessment"] for row in rows)
    assert before["answer_correct"] is False
    assert after["answer_correct"] is True
    assert before["answer_grounded"] is None and after["answer_grounded"] is None
    assert before["ordinary_aggregate_eligible"] is True
    assert after["ordinary_aggregate_eligible"] is False
    assert before["execution_errors"] == after["execution_errors"] == []
    assert completed.stderr == ""
