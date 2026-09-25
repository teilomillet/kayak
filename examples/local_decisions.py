"""Ask two questions about one input, reusing a resident model.

Install: uv sync --extra local
Run: uv run --extra local -m examples.local_decisions
Override: KAYAK_MODEL=/path/to/bundle KAYAK_DEVICE=auto

The first run downloads the full encoder. These are illustrative questions;
the printed choices are model outputs, not validated task accuracy.
"""

import os

import kayak
from kayak import Choice, DecisionResult


def questions() -> dict[str, Choice]:
    return {
        "department": Choice(
            instructions="Which team should handle this request?",
            criteria={
                "billing": "Charges, invoices, and refunds",
                "technical": "Bugs and service outages",
            },
        ),
        "urgency": Choice(
            instructions="How urgently does this request need attention?",
            criteria={
                "routine": "A question that can wait for normal support hours",
                "urgent": "A problem that prevents the customer from working",
            },
        ),
    }


def display(result: DecisionResult) -> None:
    for question_id, answer in result.answers.items():
        print(f"{question_id}: {answer.choice}")
        for candidate_id, score in answer.scores.items():
            share = answer.probabilities[candidate_id]
            print(f"  {candidate_id}: score={score:.4f}, candidate share={share:.3f}")
    print(f"Encoded tokens: {result.input_tokens}")


def main() -> None:
    with kayak.load(
        os.environ.get("KAYAK_MODEL", "Contrastive-LM/CLM-v0.1-8B"),
        device=os.environ.get("KAYAK_DEVICE", "auto"),
        cache_dir=os.environ.get("KAYAK_CACHE_DIR"),
    ) as model:
        print(f"Device: {model.info.device}; precision: {model.info.dtype}")
        print(f"Model fingerprint: {model.info.fingerprint}")
        for state in (
            "I was charged twice for my subscription.",
            "Our entire team cannot log in and work has stopped.",
        ):
            print(f"\nRequest: {state}")
            display(model.decide(state=state, questions=questions()))


if __name__ == "__main__":
    main()
