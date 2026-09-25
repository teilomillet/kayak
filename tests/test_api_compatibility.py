"""Check the frozen /v1 bodies and optional informational header policy.

The fixtures describe the contract used by 4c26ed8 and 8fbaad4. The tied scores
and first-candidate winner are hand-authored, independent of result assembly.
Adding fields to these bodies requires a new API version; unknown headers must
remain harmless. Cross-version execution lives in scripts/check_compatibility.py.
"""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from kayak import Client, DecisionRequest, TransportError

FIXTURES = Path(__file__).parent / "fixtures/api_v1"


@pytest.mark.parametrize("headers", [{}, {"X-Future-Information": "optional"}])
def test_fixed_request_and_response_remain_usable(headers: dict[str, str]) -> None:
    request = DecisionRequest.model_validate_json((FIXTURES / "request.json").read_bytes())
    response = (FIXTURES / "response.json").read_bytes()
    calls: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        assert wire.method == "POST" and wire.url.path == "/v1/decide"
        assert json.loads(wire.content) == json.loads((FIXTURES / "request.json").read_bytes())
        return httpx.Response(200, content=response, headers=headers)

    with Client(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as client:
        result = client.decide(state=request.state, questions=request.questions)
    assert json.loads(result.model_dump_json()) == json.loads(response)
    assert result.answers["department"].choice == "billing"
    assert list(result.answers["department"].scores) == ["billing", "technical"]
    assert len(calls) == 1


def test_model_info_accepts_optional_informational_headers() -> None:
    payload: JsonValue = json.loads((FIXTURES / "response.json").read_bytes())
    assert isinstance(payload, dict)
    expected = payload["model"]

    def respond(wire: httpx.Request) -> httpx.Response:
        assert wire.method == "GET" and wire.url.path == "/v1/model"
        return httpx.Response(200, json=expected, headers={"X-Future-Information": "optional"})

    with Client(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as client:
        assert client.model_info().model_dump() == expected


@pytest.mark.parametrize(
    ("operation", "path"),
    [
        ("decide", ()),
        ("decide", ("model",)),
        ("decide", ("answers", "department")),
        ("model_info", ()),
    ],
)
def test_current_client_rejects_additive_response_fields(
    operation: str, path: tuple[str, ...]
) -> None:
    payload: JsonValue = json.loads((FIXTURES / "response.json").read_bytes())
    assert isinstance(payload, dict)
    if operation == "model_info":
        payload = payload["model"]
        assert isinstance(payload, dict)
    target: JsonValue = payload
    for field in path:
        assert isinstance(target, dict)
        target = target[field]
    assert isinstance(target, dict)
    target["additional_metadata"] = "an additive server field"
    request = DecisionRequest.model_validate_json((FIXTURES / "request.json").read_bytes())
    calls: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        calls.append(wire)
        return httpx.Response(200, json=payload)

    with Client(base_url="http://fixture.test", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(TransportError):
            if operation == "model_info":
                client.model_info()
            else:
                client.decide(state=request.state, questions=request.questions)
    assert len(calls) == 1
