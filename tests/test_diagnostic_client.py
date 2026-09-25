"""Diagnostic headers enrich errors without changing response JSON or retry policy."""

import re

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kayak import Choice, Client, InputError, RemoteError, TransportError


@pytest.mark.parametrize("operation", ["decide", "model_info"])
@pytest.mark.parametrize("status", [200, 422, 503])
def test_remote_error_retains_request_id(operation: str, status: int) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            headers={"x-request-id": "r-123"},
            json={
                "error": {
                    "code": "invalid_request" if status == 422 else "overloaded",
                    "message": "failed",
                }
            },
        )

    expected = TransportError if status == 200 else InputError if status == 422 else RemoteError
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(expected) as failure:
            if operation == "model_info":
                client.model_info()
            else:
                client.decide(
                    state="s", questions={"q": Choice(instructions="i", criteria={"a": "a"})}
                )
    assert failure.value.request_id == "r-123"
    assert len(calls) == 1


@given(value=st.binary(max_size=256))
def test_untrusted_request_ids_are_bounded_printable_tokens(value: bytes) -> None:
    with Client(
        base_url="http://test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(500, headers=[(b"x-request-id", value)])
        ),
    ) as client:
        with pytest.raises(RemoteError) as failure:
            client.model_info()
    expected = value.decode("ascii") if re.fullmatch(rb"[A-Za-z0-9_.-]{1,128}", value) else None
    assert failure.value.request_id == expected


def test_no_remote_response_means_no_server_identity() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with Client(base_url="http://test", transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(TransportError) as failure:
            client.model_info()
        assert failure.value.request_id is None
        with pytest.raises(InputError) as invalid:
            client.decide(state="", questions={})
        assert invalid.value.request_id is None
    with Client(
        base_url="http://test", transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as client:
        with pytest.raises(RemoteError) as remote_failure:
            client.model_info()
        assert remote_failure.value.request_id is None
