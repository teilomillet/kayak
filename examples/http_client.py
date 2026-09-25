"""Use the same decision API through a running local service.

Install client: uv sync
Start server in another terminal: uv run --extra serve kayak serve --device auto
Run: uv run -m examples.http_client
Override: KAYAK_BASE_URL=http://127.0.0.1:8000 KAYAK_API_KEY=...

The client needs no inference libraries. This example makes one attempt;
a timeout does not establish whether remote computation has finished.
"""

import os
import sys

import kayak
from kayak import Choice, RemoteError, TransportError


def main() -> int:
    question = Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Charges, invoices, and refunds",
            "technical": "Bugs and service outages",
        },
    )
    try:
        with kayak.Client(
            base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
            api_key=os.environ.get("KAYAK_API_KEY"),
            timeout=120.0,
        ) as client:
            result = client.decide(
                state="I was charged twice for my subscription.",
                questions={"department": question},
            )
    except RemoteError as exc:
        print(f"Server error {exc.status_code} ({exc.code}): {exc}", file=sys.stderr)
        if exc.status_code == 503:
            print("The model is unavailable or busy; try again later.", file=sys.stderr)
        elif exc.status_code == 504:
            print("Computation may still be running on the server.", file=sys.stderr)
        return 1
    except TransportError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        print("Check the service address and connectivity before retrying.", file=sys.stderr)
        return 1

    answer = result.answers["department"]
    print(f"Selected department: {answer.choice}")
    print(f"Candidate shares (uncalibrated): {answer.probabilities}")
    print(f"Remote device: {result.model.device}")
    print(f"Model fingerprint: {result.model.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
