"""Exercise agent and search examples with the real client and simulated decisions."""

import math
from collections.abc import Callable

import httpx
import pytest

import kayak
from examples import rerank_documents, route_tools


def simulated_response(
    submitted: kayak.DecisionRequest, winner_index: int = 0, unknown_id: bool = False
) -> httpx.Response:
    """Assign scores independently of text; these fixtures make no quality claim."""
    answers = {}
    for question_id, question in submitted.questions.items():
        candidate_ids = list(question.criteria)
        if unknown_id:
            candidate_ids[0] = "not-submitted"
        winner = candidate_ids[winner_index]
        scores = {candidate_id: float(candidate_id == winner) for candidate_id in candidate_ids}
        total = sum(math.exp(score) for score in scores.values())
        probabilities = {
            candidate_id: math.exp(score) / total for candidate_id, score in scores.items()
        }
        answers[question_id] = kayak.ChoiceAnswer(
            choice=winner, scores=scores, probabilities=probabilities
        )
    result = kayak.DecisionResult(
        model=kayak.ModelInfo(
            id="example/simulated",
            revision="fixture-v1",
            fingerprint="simulated-no-weights",
            encoder="none",
            encoder_revision="none",
            device="none",
            dtype="none",
        ),
        answers=answers,
        input_tokens=0,
    )
    return httpx.Response(200, content=result.model_dump_json())


def recording_transport(requests: list[kayak.DecisionRequest]) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        submitted = kayak.DecisionRequest.model_validate_json(request.content)
        requests.append(submitted)
        return simulated_response(submitted)

    return httpx.MockTransport(respond)


@pytest.mark.parametrize("winner_index", [0, 1])
@pytest.mark.parametrize(
    ("main", "question_id", "candidate_ids", "state_excerpt", "expected_outputs"),
    [
        (
            route_tools.main,
            "rank",
            ("lookup_invoice", "search_help"),
            "invoice 4411",
            ("Suggested tool: lookup_invoice", "Suggested tool: search_help"),
        ),
        (
            rerank_documents.main,
            "rank",
            ("doc-plan", "doc-invoice", "doc-access"),
            "download an invoice",
            ("1. doc-plan (help/plans)", "1. doc-invoice (help/invoices)"),
        ),
    ],
)
def test_examples_submit_expected_inputs_and_follow_changed_choices(
    main: Callable[[], None],
    question_id: str,
    candidate_ids: tuple[str, ...],
    state_excerpt: str,
    expected_outputs: tuple[str, str],
    winner_index: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        submitted = kayak.DecisionRequest.model_validate_json(request.content)
        assert state_excerpt in submitted.state
        assert list(submitted.questions) == [question_id]
        assert tuple(submitted.questions[question_id].criteria) == candidate_ids
        return simulated_response(submitted, winner_index)

    original_client = kayak.Client

    def client(*, base_url: str, api_key: str | None) -> kayak.Client:
        return original_client(
            base_url=base_url, api_key=api_key, transport=httpx.MockTransport(respond)
        )

    monkeypatch.setattr(kayak, "Client", client)
    monkeypatch.setenv("KAYAK_BASE_URL", "http://example.test/api")
    monkeypatch.setenv("KAYAK_API_KEY", "example-key")
    main()
    output = capsys.readouterr().out
    assert expected_outputs[winner_index] in output
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://example.test/api/v1/decide"
    assert requests[0].headers["authorization"] == "Bearer example-key"

    if main is rerank_documents.main:
        assert "doc-plan (help/plans)\nChange your plan in Billing, under Subscription." in output
        assert "doc-invoice (help/invoices)\nOpen Billing, then Invoices." in output
        assert "doc-access (help/members)\nWorkspace owners can invite members" in output


@pytest.mark.parametrize(
    "main",
    [
        route_tools.main,
        rerank_documents.main,
    ],
)
@pytest.mark.parametrize("failure", ["busy", "unknown_id"])
def test_examples_propagate_failures_without_printing_a_choice_or_retrying(
    main: Callable[[], None],
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "busy":
            return httpx.Response(503, json={"error": {"code": "busy", "message": "Busy"}})
        submitted = kayak.DecisionRequest.model_validate_json(request.content)
        return simulated_response(submitted, unknown_id=True)

    original_client = kayak.Client

    def client(*, base_url: str, api_key: str | None) -> kayak.Client:
        return original_client(
            base_url=base_url, api_key=api_key, transport=httpx.MockTransport(respond)
        )

    monkeypatch.setattr(kayak, "Client", client)
    monkeypatch.setenv("KAYAK_BASE_URL", "http://example.test")
    monkeypatch.delenv("KAYAK_API_KEY", raising=False)
    expected_error = kayak.RemoteError if failure == "busy" else kayak.TransportError
    with pytest.raises(expected_error):
        main()
    assert len(requests) == 1
    assert capsys.readouterr().out == ""


def test_tool_permissions_are_applied_before_inference() -> None:
    requests: list[kayak.DecisionRequest] = []
    tools = {"read": "Read invoice", "delete": "Delete invoice", "search": "Search help"}
    with kayak.Client(
        base_url="http://example.test", transport=recording_transport(requests)
    ) as client:
        selected = route_tools.select_tool(
            client, "Inspect invoice", tools, frozenset({"read", "search", "unknown"})
        )
        assert selected == "read"
        assert requests[0].questions["rank"].criteria == {
            "read": "Read invoice",
            "search": "Search help",
        }
        assert route_tools.select_tool(client, "Inspect invoice", tools, frozenset()) is None
        assert (
            route_tools.select_tool(client, "Inspect invoice", tools, frozenset({"unknown"}))
            is None
        )
    assert len(requests) == 1
    assert list(tools) == ["read", "delete", "search"]
