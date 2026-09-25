"""Call a resident service from async application code, one decision at a time.

Run: uv run -m examples.async_decisions
Service: uv run --extra serve kayak serve --device auto
Configure: KAYAK_BASE_URL and KAYAK_API_KEY

The application supplies candidate actions. This program prints the ranking;
it does not execute tools. Async HTTP lets other application tasks progress
while waiting; the service still admits one inference and has no request queue.
"""

import asyncio
import os

from kayak import AsyncClient, InputError, RemoteError, TransportError

ACTIONS = {
    "lookup_invoice": "Look up the invoice and payment history to investigate a duplicate charge.",
    "reset_password": "Send a password reset link to restore account access.",
    "ask_customer": "Ask the customer for the missing details before taking an action.",
}


async def main() -> int:
    try:
        async with AsyncClient(
            base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
            api_key=os.environ.get("KAYAK_API_KEY"),
        ) as client:
            for state in ("I was charged twice for invoice 4411.", "Can you help with my account?"):
                result = await client.rank(
                    state=state,
                    instructions="Which action should the support application consider next?",
                    candidates=ACTIONS,
                )
                print(result.model_dump_json(), flush=True)
    except (InputError, RemoteError, TransportError) as exc:
        import sys

        print(f"{type(exc).__name__}: {exc}; no automatic retry", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
