"""Exercise application code without a model, server, network, or GPU.

Install: uv sync
Run: uv run -m examples.mock_integration

This deliberately simulated response tests a consumer's handling of typed
decisions. It says nothing about model quality. The public HTTP transport hook
lets tests use the real client, including its response validation.
"""

import httpx

import kayak
from kayak import Choice, ChoiceAnswer, DecisionResult, ModelInfo


def simulated_response(request: httpx.Request) -> httpx.Response:
    if request.method != "POST" or request.url.path != "/v1/decide":
        return httpx.Response(404)
    result = DecisionResult(
        model=ModelInfo(
            id="example/simulated",
            revision="fixture-v1",
            fingerprint="simulated-no-weights",
            encoder="none",
            encoder_revision="none",
            device="none",
            dtype="none",
        ),
        answers={
            "department": ChoiceAnswer(
                choice="billing",
                scores={"billing": 0.0, "technical": 0.0},
                probabilities={"billing": 0.5, "technical": 0.5},
            )
        },
        input_tokens=0,
    )
    return httpx.Response(200, content=result.model_dump_json())


def main() -> None:
    question = Choice(
        instructions="Which team should handle this request?",
        criteria={"billing": "Charges and refunds", "technical": "Bugs and outages"},
    )
    with kayak.Client(
        base_url="http://example.test",
        transport=httpx.MockTransport(simulated_response),
    ) as client:
        result = client.decide(state="I was charged twice.", questions={"department": question})
    assert result.answers["department"].choice == "billing"
    assert result.calibration == "none"
    print("Simulated integration passed; no model was loaded.")


if __name__ == "__main__":
    main()
