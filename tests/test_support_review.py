"""Independent review decisions, input integrity, and pre-inference dataset challenges."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import httpx
import pytest

from examples import evaluate_support, prepare_support
from examples.evaluate_use_cases import simulated_response
from examples.support_data import (
    RESOLUTION_COLUMNS,
    REVIEW_COLUMNS,
    Packet,
    Source,
    audit_support,
    compile_reviews,
    csv_bytes,
    csv_rows,
    json_bytes,
    load_prepared,
    read_record,
    resolution_rows,
    resolve_reviews,
    review_sheet,
)
from kayak.eval import load_report, load_suite

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("billing", "shipping", "account", "review", "billing", "shipping", "account", "review")


def packet_fixture() -> Packet:
    return Packet(
        source=read_record((ROOT / "examples/support_review/tickets.json").read_bytes(), Source),
        question=load_suite(ROOT / "examples/suites/support_pilot.json").question,
        reviewers=["reviewer-a", "reviewer-b"],
    )


def completed_reviews(packet: Packet, *, disagreement: bool = False) -> list[bytes]:
    files = []
    for reviewer in packet.reviewers:
        rows = csv_rows(review_sheet(packet, reviewer), REVIEW_COLUMNS)
        for row, label in zip(rows, LABELS, strict=True):
            row["label"] = label
            row["notes"] = "Fictional annotation; café, multiline\nnotes are preserved."
        if disagreement and reviewer == packet.reviewers[0]:
            rows[-1]["label"] = "billing"
        files.append(csv_bytes(REVIEW_COLUMNS, rows))
    return files


def completed_resolution(packet: Packet, reviews: list[bytes]) -> bytes:
    rows = resolution_rows(packet, reviews)
    for row in rows:
        row.update(
            label="review", adjudicator="review-owner", reason="Two different teams' issues."
        )
    return csv_bytes(RESOLUTION_COLUMNS, rows)


def prepared_fixture(tmp_path: Path, packet: Packet | None = None) -> Path:
    packet = packet or packet_fixture()
    (tmp_path / "packet.json").write_bytes(json_bytes(packet.model_dump(mode="json")))
    reviews = completed_reviews(packet, disagreement=True)
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for path, raw in zip(paths, reviews, strict=True):
        path.write_bytes(raw)
    adjudications = tmp_path / "resolved.csv"
    adjudications.write_bytes(completed_resolution(packet, reviews))
    destination = tmp_path / "prepared"
    prepare_support.import_reviews(tmp_path / "packet.json", paths, adjudications, destination)
    return destination


def test_export_contains_no_labels_and_binds_exact_text_and_question(tmp_path: Path) -> None:
    destination = tmp_path / "packet"
    prepare_support.export_packet(
        ROOT / "examples/support_review/tickets.json",
        ROOT / "examples/suites/support_pilot.json",
        ["reviewer-a", "reviewer-b"],
        destination,
    )
    packet = read_record((destination / "packet.json").read_bytes(), Packet)
    assert packet.source.source_kind == "fictional"
    assert packet.question == load_suite(ROOT / "examples/suites/support_pilot.json").question
    assert "label" not in packet.source.model_dump_json()
    for index, reviewer in enumerate(packet.reviewers, 1):
        rows = csv_rows((destination / f"review-{index}.csv").read_bytes(), REVIEW_COLUMNS)
        assert [row["id"] for row in rows] == [ticket.id for ticket in packet.source.tickets]
        for row, ticket in zip(rows, packet.source.tickets, strict=True):
            assert row["label"] == row["notes"] == ""
            assert row["reviewer"] == reviewer
            assert row["packet_sha256"] == packet.sha256
            assert json.loads(row["text_json"]) == ticket.text


def test_spreadsheet_text_cells_are_quoted_without_modifying_ticket_content() -> None:
    packet = packet_fixture()
    packet.source.tickets[0].text = '=HYPERLINK("somewhere", "click")\n+not a formula'
    rows = csv_rows(review_sheet(packet, packet.reviewers[0]), REVIEW_COLUMNS)
    assert rows[0]["text_json"].startswith('"')
    assert json.loads(rows[0]["text_json"]) == packet.source.tickets[0].text


def test_long_valid_json_quoted_ticket_survives_csv_exchange() -> None:
    packet = packet_fixture()
    # The request permits 65,536 characters. JSON quoting doubles backslashes,
    # taking a valid ticket above Python's default 131,072-character CSV field cap.
    packet.source.tickets[0].text = "\\" * 65_536
    packet = Packet.model_validate(packet.model_dump(mode="json"))
    previous_limit = csv.field_size_limit()
    reviews = completed_reviews(packet)
    suites, _, _ = compile_reviews(packet, reviews, None)
    assert suites[0].examples[0].text == packet.source.tickets[0].text
    assert csv.field_size_limit() == previous_limit


def test_disagreement_needs_explicit_resolution_and_preserves_both_votes() -> None:
    packet = packet_fixture()
    reviews = completed_reviews(packet, disagreement=True)
    with pytest.raises(ValueError, match="unresolved"):
        compile_reviews(packet, reviews, None)
    rows = resolution_rows(packet, reviews)
    assert len(rows) == 1
    assert rows[0]["id"] == "r008"
    assert (rows[0]["first_label"], rows[0]["second_label"]) == ("billing", "review")
    suites, receipt, audit = compile_reviews(packet, reviews, completed_resolution(packet, reviews))
    assert [suite.split for suite in suites] == ["development", "test"]
    assert [row.label for suite in suites for row in suite.examples] == list(LABELS)
    assert [row.text for suite in suites for row in suite.examples] == [
        row.text for row in packet.source.tickets
    ]
    decisions = resolve_reviews(packet, reviews, completed_resolution(packet, reviews))
    assert decisions[-1].adjudicator == "review-owner"
    assert [vote.label for vote in decisions[-1].votes] == ["billing", "review"]
    assert decisions[-1].votes[0].notes.endswith("\nnotes are preserved.")
    assert receipt["source_kind"] == "fictional"
    assert audit["status"] == "clear"
    assert audit["grouping_checked"] is audit["cross_split_checked"] is True


@pytest.mark.parametrize(
    "change", ["text", "hash", "id", "missing", "duplicate", "label", "reviewer"]
)
def test_changed_or_incomplete_review_is_rejected(change: str) -> None:
    packet = packet_fixture()
    reviews = completed_reviews(packet)
    rows = csv_rows(reviews[0], REVIEW_COLUMNS)
    if change == "text":
        rows[0]["text_json"] = json.dumps("Changed ticket")
    elif change == "hash":
        rows[0]["packet_sha256"] = "0" * 64
    elif change == "id":
        rows[0]["id"] = "unknown"
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "label":
        rows[0]["label"] = ""
    else:
        rows[0]["reviewer"] = "someone-else"
    reviews[0] = csv_bytes(REVIEW_COLUMNS, rows)
    with pytest.raises(ValueError):
        compile_reviews(packet, reviews, None)


def test_adjudication_cannot_be_reused_after_reviews_change() -> None:
    packet = packet_fixture()
    reviews = completed_reviews(packet, disagreement=True)
    resolutions = completed_resolution(packet, reviews)
    rows = csv_rows(reviews[0], REVIEW_COLUMNS)
    rows[0]["notes"] = "A different observation"
    reviews[0] = csv_bytes(REVIEW_COLUMNS, rows)
    with pytest.raises(ValueError, match="inputs changed"):
        compile_reviews(packet, reviews, resolutions)


@pytest.mark.parametrize("change", ["blank_reason", "blank_owner", "extra", "missing"])
def test_adjudication_requires_exact_disputes_and_a_recorded_decision(change: str) -> None:
    packet = packet_fixture()
    reviews = completed_reviews(packet, disagreement=True)
    rows = csv_rows(completed_resolution(packet, reviews), RESOLUTION_COLUMNS)
    if change == "blank_reason":
        rows[0]["reason"] = " "
    elif change == "blank_owner":
        rows[0]["adjudicator"] = ""
    elif change == "extra":
        rows.append(dict(rows[0], id="r001"))
    else:
        rows.clear()
    with pytest.raises(ValueError):
        compile_reviews(packet, reviews, csv_bytes(RESOLUTION_COLUMNS, rows))


def test_text_group_and_identifier_overlap_are_visible_without_changing_data() -> None:
    packet = packet_fixture()
    packet.source.tickets[4].text = "  MY CARD was charged twice for one purchase.  "
    packet.source.tickets[4].group_id = packet.source.tickets[0].group_id
    reviews = completed_reviews(packet)
    suites, _, audit = compile_reviews(packet, reviews, None)
    before = [suite.model_dump_json() for suite in suites]
    assert audit["status"] == "needs_review"
    assert '"group_overlap"' in json.dumps(audit)
    assert '"casefold_whitespace"' in json.dumps(audit)
    suites[1].examples[0].label = "review"
    suites[1].examples[0].id = suites[0].examples[0].id
    report = audit_support(suites)
    assert '"conflicting_labels": true' in json.dumps(report)
    assert '"id_overlap"' in json.dumps(report)
    assert before[0] == suites[0].model_dump_json()


@pytest.mark.parametrize(
    "filename", ["test.json", "audit.json", "review-record.json", "packet.json"]
)
def test_prepared_files_are_reverified_from_raw_reviews(tmp_path: Path, filename: str) -> None:
    directory = prepared_fixture(tmp_path)
    suites, audit = load_prepared(directory)
    assert len(suites) == 2 and audit["status"] == "clear"
    path = directory / filename
    payload = json.loads(path.read_bytes())
    if filename == "test.json":
        payload["examples"][0]["label"] = "review"
    elif filename == "audit.json":
        payload["status"] = "needs_review"
    elif filename == "review-record.json":
        payload["decisions"][0]["reason"] = "Altered"
    else:
        payload["question"]["instructions"] += " Altered"
    path.write_bytes(json_bytes(payload))
    with pytest.raises(ValueError):
        load_prepared(directory)


def test_existing_output_and_invalid_source_never_get_replaced(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep")
    assert (
        prepare_support.main(
            [
                "export",
                str(ROOT / "examples/support_review/tickets.json"),
                "--reviewers",
                "a",
                "b",
                "--output",
                str(existing),
            ]
        )
        == 2
    )
    assert marker.read_text() == "keep"
    source = json.loads((ROOT / "examples/support_review/tickets.json").read_bytes())
    source["tickets"][0]["label"] = "billing"
    path = tmp_path / "invalid.json"
    path.write_bytes(json_bytes(source))
    output = tmp_path / "absent"
    assert (
        prepare_support.main(
            [
                "export",
                str(path),
                "--reviewers",
                "a",
                "b",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_prepared_simulation_sends_only_text_and_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_fixture(tmp_path)
    captured: list[dict[str, object]] = []

    def record(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return simulated_response(request)

    monkeypatch.setattr(evaluate_support, "simulated_response", record)
    output = tmp_path / "simulation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_support",
            "--prepared",
            str(prepared),
            "--simulate",
            "--output",
            str(output),
        ],
    )
    assert evaluate_support.main() == 0
    assert len(captured) == 5  # One warmup and the four held-out fixtures.
    assert all(set(row) == {"state", "questions"} for row in captured)
    assert load_report(output).status == "complete"
    assessment = json.loads((output / "acceptance.json").read_bytes())
    assert assessment["deployment_accepted"] is False
    assert assessment["provisional_gates_passed"] is False
    assert assessment["evidence_scope"] == "integration_only"
    audit = json.loads((output / "dataset-audit.json").read_bytes())
    assert audit["grouping_checked"] is audit["cross_split_checked"] is True


def test_leaked_conversation_prevents_client_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = packet_fixture()
    packet.source.tickets[4].group_id = packet.source.tickets[0].group_id
    prepared = prepared_fixture(tmp_path, packet)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("a client must not be created before resolving dataset findings")

    monkeypatch.setattr(evaluate_support, "Client", forbidden)
    output = tmp_path / "blocked-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_support",
            "--prepared",
            str(prepared),
            "--output",
            str(output),
        ],
    )
    assert evaluate_support.main() == 1
    assert (output / "dataset-audit.json").exists()
    assert not (output / "report.json").exists()


@pytest.mark.parametrize("failure", ["timeout", "interrupt"])
def test_prepared_workflow_retains_failures_and_preflight(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_fixture(tmp_path)
    calls = 0
    failed_text = ""

    def fail_one(request: httpx.Request) -> httpx.Response:
        nonlocal calls, failed_text
        calls += 1
        if calls == 2:
            failed_text = json.loads(request.content)["state"]
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise httpx.ReadTimeout("controlled timeout", request=request)
        return simulated_response(request)

    monkeypatch.setattr(evaluate_support, "simulated_response", fail_one)
    output = tmp_path / "failed-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_support",
            "--prepared",
            str(prepared),
            "--simulate",
            "--output",
            str(output),
        ],
    )
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            evaluate_support.main()
    else:
        assert evaluate_support.main() == 1
        assert calls == 5  # A warmup and four attempts, with no retry after uncertainty.
        assessment = json.loads((output / "acceptance.json").read_bytes())
        failed_id = next(
            row.id for row in load_suite(prepared / "test.json").examples if row.text == failed_text
        )
        outcome = next(row for row in assessment["outcomes"] if row["id"] == failed_id)
        assert outcome["suggestion"] is None
        assert outcome["review_queue"] == "review"
        assert assessment["deployment_accepted"] is False
    report = load_report(output)
    assert report.status == ("interrupted" if failure == "interrupt" else "failed")
    saved = json.loads(report.model_dump_json())
    assert saved["config"]["dataset_audit"]["grouping_checked"] is True
    assert saved["config"]["dataset_audit"]["cross_split_checked"] is True


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_record(b'{"name": "first", "name": "second"}', Source)
