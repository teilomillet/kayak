"""Probe a real service using one isolated, installed SDK and fixed expectations."""

import argparse
import asyncio
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import monotonic, sleep

import httpx

import kayak
from kayak import Client, DecisionRequest, InputError, RemoteError


class Arguments(argparse.Namespace):
    url: str
    fixtures: Path
    work: Path
    extensions: bool


class CountingTransport(httpx.HTTPTransport):
    """Observe SDK attempts while still using real sockets and HTTP parsing."""

    calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return super().handle_request(request)


def expect_remote_error(client: Client, request: DecisionRequest, status: int, code: str) -> None:
    try:
        client.decide(state=request.state, questions=request.questions)
    except RemoteError as exc:
        assert exc.status_code == status and exc.code == code, (exc.status_code, exc.code)
    else:
        raise AssertionError(f"expected HTTP {status} ({code})")


async def check_extensions(url: str, request: DecisionRequest) -> None:
    # Only the current installation is asked to expose new Python interfaces.
    # Both installed server versions must accept their unchanged /v1 requests.
    from kayak import AsyncClient, JudgmentQuestion, Noul, NoulAnswer, Score, ScoreAnswer

    question = request.questions["department"]
    judgments: dict[str, JudgmentQuestion] = {
        "noul": Noul(instructions="The customer needs help."),
        "score": Score(instructions="Assess impact", criteria=["Low", "Medium", "High"]),
    }
    with Client(base_url=url, api_key="compatibility-fixture") as client:
        expected = client.decide(state=request.state, questions=request.questions)
        ranked = client.rank(
            state=request.state, instructions=question.instructions, candidates=question.criteria
        )
        judged = client.judge(state=request.state, questions=judgments)
        binary, score = judged.answers["noul"], judged.answers["score"]
        assert isinstance(binary, NoulAnswer) and binary.noul == 0.5
        assert isinstance(score, ScoreAnswer) and score.score == 1.0
        assert score.legend == {"0": "Low", "1": "Medium", "2": "High"}
    assert [item.id for item in ranked.ranked] == ["billing", "technical"]
    assert [item.probability for item in ranked.ranked] == [0.5, 0.5]
    async with AsyncClient(base_url=url, api_key="compatibility-fixture") as client:
        assert await client.judge(state=request.state, questions=judgments) == judged
        assert await client.model_info() == expected.model
        assert await client.decide(state=request.state, questions=request.questions) == expected
        assert (
            await client.rank(
                state=request.state,
                instructions=question.instructions,
                candidates=question.criteria,
            )
            == ranked
        )
        try:
            await client.decide(state="test:invalid", questions=request.questions)
        except InputError:
            pass
        else:
            raise AssertionError("expected async InputError")
        assert await client.decide(state=request.state, questions=request.questions) == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--extensions", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    assert sys.flags.isolated and not sys.flags.optimize, "run with -I, without -O"
    assert kayak.__file__ is not None
    package = Path(kayak.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve()), "install Kayak without -e"
    request = DecisionRequest.model_validate_json((args.fixtures / "request.json").read_bytes())
    expected: object = json.loads((args.fixtures / "response.json").read_bytes())
    assert isinstance(expected, dict)
    checks: list[str] = []
    with httpx.Client(base_url=args.url, timeout=2, trust_env=False) as http:
        deadline = monotonic() + 10
        while True:
            try:
                response = http.get("/health")
                assert response.status_code == 200 and response.json() == {"ready": True}
                break
            except httpx.ConnectError:
                assert monotonic() < deadline, "server did not become ready"
                sleep(0.02)
        checks.append("readiness")
        for content, status, code in (
            (b"{", 422, "invalid_request"),
            (b"x" * (1_048_576 + 1), 413, "request_too_large"),
        ):
            response = http.post(
                "/v1/decide",
                content=content,
                headers={"authorization": "Bearer compatibility-fixture"},
            )
            payload: object = response.json()
            assert response.status_code == status and isinstance(payload, dict)
            assert set(payload) == {"error"} and isinstance(payload["error"], dict)
            assert set(payload["error"]) == {"code", "message"}
            assert payload["error"]["code"] == code
            assert isinstance(payload["error"]["message"], str)
            checks.append(code)

    transport = CountingTransport()
    with Client(base_url=args.url, transport=transport) as unauthorized:
        expect_remote_error(unauthorized, request, 401, "unauthorized")
        assert transport.calls == 1, "SDK retried an unauthorized call"
    checks.append("unauthorized")

    transport = CountingTransport()
    with Client(
        base_url=args.url,
        api_key="compatibility-fixture",
        timeout=3,
        transport=transport,
    ) as client:
        assert client.model_info().model_dump() == expected["model"]
        checks.append("model_info")
        for questions in (
            request.questions,
            {name: choice.model_dump() for name, choice in request.questions.items()},
        ):
            result = client.decide(state=request.state, questions=questions)
            assert result.model_dump() == expected
            answer = result.answers["department"]
            assert list(answer.scores) == list(answer.probabilities) == ["billing", "technical"]
            assert answer.choice == "billing", "ties must select the first candidate"
        checks.append("typed_and_dictionary_decisions")
        before = transport.calls
        for state in (" ", "test:invalid"):
            try:
                client.decide(state=state, questions=request.questions)
            except InputError:
                pass
            else:
                raise AssertionError("expected InputError")
        assert transport.calls == before + 1, "blank input must be local; remote 422 is one attempt"
        checks.append("local_and_remote_input_errors")
        for state, status, code in (
            ("test:failed", 500, "inference_failed"),
            ("test:closed", 503, "not_ready"),
            ("test:block", 504, "timeout"),
        ):
            before = transport.calls
            controlled = DecisionRequest(state=state, questions=request.questions)
            expect_remote_error(client, controlled, status, code)
            assert transport.calls == before + 1, f"SDK retried {code}"
            checks.append(code)
        before = transport.calls
        expect_remote_error(client, request, 503, "overloaded")
        assert transport.calls == before + 1, "SDK retried overload"
        checks.append("timeout_retains_capacity")
        (args.work / "release").touch()
        deadline = monotonic() + 5
        while True:
            # These bounded polls belong to the test; the SDK itself must not retry.
            before = transport.calls
            try:
                result = client.decide(state=request.state, questions=request.questions)
                assert result.model_dump() == expected
                break
            except RemoteError as exc:
                assert exc.status_code == 503 and exc.code == "overloaded"
                assert monotonic() < deadline, "capacity did not recover"
                sleep(0.02)
            finally:
                assert transport.calls == before + 1
        checks.append("capacity_recovery")
    if args.extensions:
        asyncio.run(check_extensions(args.url, request))
        checks.append("ranking_judgments_and_async_extensions")
    assert not {"torch", "transformers", "huggingface_hub"} & sys.modules.keys()
    print(
        json.dumps(
            {
                "package": str(package),
                "python": sys.version,
                "versions": {name: version(name) for name in ("kayak", "pydantic", "httpx")},
                "checks": checks,
            }
        )
    )


if __name__ == "__main__":
    main()
