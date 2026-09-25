"""Ask Noul and Score questions through an existing Kayak service.

Install: uv sync
Service: uv run --extra serve kayak serve --device auto
Run: uv run -m examples.typed_judgments

KAYAK_BASE_URL and KAYAK_API_KEY configure the client. Outputs are uncalibrated
candidate shares and a rubric average; the example takes no application action.
"""

import os

from kayak import Client, JudgmentQuestion, Noul, NoulAnswer, Score, ScoreAnswer


def main() -> None:
    questions: dict[str, JudgmentQuestion] = {
        "duplicate_charge": Noul(instructions="The customer reports being charged twice."),
        "impact": Score(
            instructions="Assess the impact described by the customer.",
            criteria=[
                "An information request with no reported disruption.",
                "A problem affecting one payment or account operation.",
                "A problem preventing all use of the account.",
            ],
        ),
    }
    with Client(
        base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
        api_key=os.environ.get("KAYAK_API_KEY"),
    ) as client:
        result = client.judge(
            state="I was charged twice for one order. I can still use my account.",
            questions=questions,
        )
    binary = result.answers["duplicate_charge"]
    impact = result.answers["impact"]
    assert isinstance(binary, NoulAnswer) and isinstance(impact, ScoreAnswer)
    print(f"True-candidate share (uncalibrated): {binary.noul:.3f}")
    print(f"Expected impact level (0–2): {impact.score:.3f}")
    print(result.model_dump_json(indent=2))  # Retain distributions, rubric, and model identity.


if __name__ == "__main__":
    main()
