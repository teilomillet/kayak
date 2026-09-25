"""Select a permitted tool for an agent without executing it.

Install client: uv sync
Start server in another terminal: uv run --extra serve kayak serve --device auto
Run: uv run -m examples.route_tools
Override: KAYAK_BASE_URL=http://127.0.0.1:8000 KAYAK_API_KEY=...

The application filters permission IDs before ranking. The result is a suggested
tool ID, not authorization or generated arguments. Rankings are uncalibrated
comparisons; evaluate tool selection on your own requests before using it.
"""

import os
from collections.abc import Mapping

import kayak


def select_tool(
    client: kayak.Client,
    state: str,
    tools: Mapping[str, str],
    permitted_tool_ids: frozenset[str],
) -> str | None:
    """Return a permitted tool ID, or None without inference if none is permitted."""
    candidates = {}
    for tool_id, description in tools.items():
        if tool_id in permitted_tool_ids:
            candidates[tool_id] = description
    if not candidates:
        return None

    result = client.rank(
        state=state,
        instructions="Which available tool would help answer this request?",
        candidates=candidates,
    )
    return result.ranked[0].id


def main() -> None:
    state = "Why was invoice 4411 charged twice? Please inspect the payment history."
    tools = {
        "lookup_invoice": "Read the invoice and its payment history.",
        "search_help": "Find help articles about billing and account settings.",
        "issue_refund": "Issue a refund for a payment.",
    }
    # These permissions come from application policy, never from user text or scores.
    permitted_tool_ids = frozenset({"lookup_invoice", "search_help"})
    base_url = os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000")
    api_key = os.environ.get("KAYAK_API_KEY")

    with kayak.Client(base_url=base_url, api_key=api_key) as client:
        selected_id = select_tool(client, state, tools, permitted_tool_ids)

    if selected_id is None:
        print("No permitted tool is available.")
        return
    print(f"Suggested tool: {selected_id}")
    print(tools[selected_id])


if __name__ == "__main__":
    main()
