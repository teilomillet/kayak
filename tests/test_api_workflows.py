"""Exercise a user's local-to-HTTP workflow with actual, randomly initialized Qwen3.

This checks API behavior and recovery, not learned decision quality.
"""

import asyncio
import socket
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
import uvicorn

from kayak import (
    AsyncClient,
    Choice,
    Client,
    DecisionRequest,
    DecisionResult,
    InputError,
    ModelClosedError,
    load,
)
from kayak.server import create_app


@pytest.mark.inference
@pytest.mark.parametrize("input_style", ["typed", "dictionary"])
def test_decide_switch_to_http_and_recover(tiny_bundle: Path, input_style: str) -> None:
    fixture = Path(__file__).parent / "fixtures/api_v1/request.json"
    request = DecisionRequest.model_validate_json(fixture.read_bytes())
    original = request.model_dump_json()
    questions: Mapping[str, Choice | Mapping[str, object]] = request.questions
    if input_style == "dictionary":
        questions = {name: question.model_dump() for name, question in request.questions.items()}
    # Valid schema, but too many tokens once the loaded tokenizer processes it.
    oversized_state = "charged " * 2048
    with load(tiny_bundle, device="cpu", batch_size=4) as model:
        with pytest.raises(InputError):
            model.decide(state=" ", questions=questions)
        with pytest.raises(InputError, match="tokens"):
            model.decide(state=oversized_state, questions=questions)
        direct = model.decide(state=request.state, questions=questions)
        assert direct.input_tokens > 0
        assert list(direct.answers) == list(request.questions)
        assert list(direct.answers["department"].scores) == ["billing", "technical"]

        server = uvicorn.Server(
            uvicorn.Config(create_app(lambda: model, api_key="workflow-test"), log_level="error")
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            address = f"http://127.0.0.1:{sock.getsockname()[1]}"
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while not server.started:
                    assert thread.is_alive() and time.monotonic() < deadline
                    time.sleep(0.01)
                with Client(base_url=address, api_key="workflow-test", timeout=5) as client:
                    assert client.model_info() == model.info
                    # The first error is local validation; the second crosses HTTP
                    # and must leave server capacity available for the corrected call.
                    with pytest.raises(InputError):
                        client.decide(state=" ", questions=questions)
                    with pytest.raises(InputError, match="tokens"):
                        client.decide(state=oversized_state, questions=questions)
                    remote = client.decide(state=request.state, questions=questions)
                    assert remote == direct
                    ranked = client.rank(
                        state=request.state,
                        instructions=request.questions["department"].instructions,
                        candidates=request.questions["department"].criteria,
                    )

                async def async_workflow() -> None:
                    async with AsyncClient(
                        base_url=address, api_key="workflow-test", timeout=5
                    ) as client:
                        assert await client.model_info() == model.info
                        with pytest.raises(InputError, match="tokens"):
                            await client.decide(state=oversized_state, questions=questions)
                        assert (
                            await client.decide(state=request.state, questions=questions) == direct
                        )
                        assert (
                            await client.rank(
                                state=request.state,
                                instructions=request.questions["department"].instructions,
                                candidates=request.questions["department"].criteria,
                            )
                            == ranked
                        )

                asyncio.run(async_workflow())
            finally:
                server.should_exit = True
                thread.join(timeout=10)
                assert not thread.is_alive(), "service did not drain and close"

    # Callers retain both their inputs and usable results after closing resources.
    assert request.model_dump_json() == original
    assert DecisionResult.model_validate_json(remote.model_dump_json()) == direct
    with pytest.raises(ModelClosedError):
        model.decide(state=request.state, questions=questions)
