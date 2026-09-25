"""Check partition policy separately from pinned-byte download integrity."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from kayak.eval import _banking77 as data
from kayak.eval import banking77, prepare_banking77


@pytest.fixture
def synthetic_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic rows match published counts; they are not language-quality evidence."""
    labels = [f"intent_{index:02d}" for index in range(77)]
    contents = {"categories.json": json.dumps(labels).encode(), "LICENSE": b"fixture"}
    for split, count in (("train", 10003), ("test", 3080)):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["text", "category"])
        for index in range(count):
            writer.writerow([f"{split} example {index}", labels[index % 77]])
        contents[f"{split}.csv"] = stream.getvalue().encode()
    monkeypatch.setattr(data, "_verified", lambda path: contents[path.name])


def test_official_counts_and_disjoint_development_partition(synthetic_source: None) -> None:
    train = banking77(split="train")
    dev = banking77(split="dev")
    test = banking77(split="test")
    assert len(train.examples) == 9233
    assert len(dev.examples) == 770
    assert len(test.examples) == 3080
    assert set(Counter(row.label for row in dev.examples).values()) == {10}
    assert set(Counter(row.label for row in test.examples).values()) == {40}
    train_ids = {row.id for row in train.examples}
    dev_ids = {row.id for row in dev.examples}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids | dev_ids == {f"train:{index}" for index in range(10003)}
    assert {row.id for row in test.examples} == {f"test:{index}" for index in range(3080)}
    assert banking77(split="dev") == dev


def test_subset_keeps_all_candidates_and_has_balanced_selection(synthetic_source: None) -> None:
    suite = banking77(limit=154)
    assert len(suite.question.criteria) == 77
    assert len(suite.examples) == 154
    assert set(Counter(row.label for row in suite.examples).values()) == {2}
    assert suite.provenance["coverage"] == "subset"
    assert suite.question.criteria["intent_00"] == "intent 00"
    assert suite.sha256 != banking77(limit=77).sha256
    with pytest.raises(ValueError, match="limit exceeds"):
        banking77(limit=771)


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_limit_fails_before_cache_access(limit: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        banking77(limit=limit, cache_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_missing_cache_is_offline_and_actionable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prepare banking77"):
        banking77(cache_dir=tmp_path)


def test_download_checks_pinned_bytes_and_corrupt_cache_is_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'["a", "b"]'
    monkeypatch.setattr(
        data, "FILES", {"categories.json": (len(raw), hashlib.sha256(raw).hexdigest())}
    )
    urls: list[str] = []

    def stream(method: str, url: str, *, timeout: int) -> httpx.Response:
        assert method == "GET" and timeout == 60
        urls.append(url)
        return httpx.Response(200, content=raw, request=httpx.Request(method, url))

    # Response is not itself a stream context manager; use a real client transport.
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: stream(request.method, str(request.url), timeout=60)
        )
    )
    monkeypatch.setattr(httpx, "stream", client.stream)
    try:
        directory = prepare_banking77(tmp_path)
        assert (directory / "categories.json").read_bytes() == raw
        assert data.REVISION in urls[0]
        assert prepare_banking77(tmp_path) == directory
        assert len(urls) == 1
        (directory / "categories.json").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="integrity"):
            prepare_banking77(tmp_path)
        assert len(urls) == 1
        assert (directory / "categories.json").read_bytes() == b"corrupt"
    finally:
        client.close()


def test_wrong_download_is_never_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "FILES", {"categories.json": (3, hashlib.sha256(b"yes").hexdigest())})
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"bad"))
    ) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        with pytest.raises(ValueError, match="integrity"):
            prepare_banking77(tmp_path)
    assert not list(tmp_path.rglob("categories.json"))
