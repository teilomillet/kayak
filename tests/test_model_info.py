"""Remote model inspection shares authentication, error, and transport contracts."""

import httpx
import pytest
from test_contract import sample_result

from kayak import Client, InputError, RemoteError, TransportError


def test_inspect_model_preserves_base_path_and_does_not_post_a_decision() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        assert request.url.path == "/kayak/v1/model"
        assert request.content == b""
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(200, content=sample_result().model.model_dump_json())

    with Client(
        base_url="https://example.test/kayak",
        api_key="key",
        transport=httpx.MockTransport(respond),
    ) as client:
        assert client.model_info() == sample_result().model
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"null", b"{}", b"[]", b"invalid", b"\xff"])
def test_model_info_rejects_malformed_payloads(body: bytes) -> None:
    with Client(
        base_url="http://example.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
    ) as client:
        with pytest.raises(TransportError, match="invalid model information"):
            client.model_info()


def test_model_info_never_follows_redirects() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://elsewhere.test"})

    with Client(base_url="http://example.test", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(RemoteError) as exc:
            client.model_info()
    assert exc.value.status_code == 307 and len(calls) == 1


@pytest.mark.parametrize("url", ["http://host:bad-port", "http://host/\nprivate"])
def test_invalid_url_is_a_kayak_input_error(url: str) -> None:
    with pytest.raises(InputError, match="valid HTTP") as exc:
        Client(base_url=url)
    assert "private" not in str(exc.value)


def test_non_ascii_api_key_is_a_private_input_error() -> None:
    with pytest.raises(InputError, match="ASCII") as exc:
        Client(base_url="http://example.test", api_key="private-é")
    assert "private" not in str(exc.value)
