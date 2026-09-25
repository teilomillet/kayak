"""Retry a busy service at most twice; propagate uncertain timeout outcomes.

Install: uv sync
Start server: uv run --extra serve kayak serve --device auto
Run: uv run -m examples.retry_busy
Override: KAYAK_BASE_URL=http://127.0.0.1:8000 KAYAK_API_KEY=...

This caller-owned policy retries only HTTP 503 with code overloaded, meaning no
inference was admitted. A transport failure or HTTP 504 may leave work running;
neither is retried. The short fixed backoff suits this single demonstration
caller; coordinate admission in your application for many concurrent workers.
"""

import os
from time import sleep

import kayak


def decide_when_available(
    client: kayak.Client, *, state: str, questions: dict[str, kayak.Choice]
) -> kayak.DecisionResult:
    retries_left = 2
    delay_seconds = 0.25
    while True:
        try:
            return client.decide(state=state, questions=questions)
        except kayak.RemoteError as exc:
            if exc.status_code != 503 or exc.code != "overloaded" or retries_left == 0:
                raise
            sleep(delay_seconds)
            retries_left -= 1
            delay_seconds *= 2


def main() -> None:
    question = kayak.Choice(
        instructions="Which team should handle this request?",
        criteria={"billing": "Charges and refunds", "technical": "Bugs and service outages"},
    )
    with kayak.Client(
        base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
        api_key=os.environ.get("KAYAK_API_KEY"),
    ) as client:
        result = decide_when_available(
            client, state="I was charged twice.", questions={"department": question}
        )
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
