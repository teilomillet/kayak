"""Synthetic pinned inputs exercise HINT3 boundaries without bundling upstream data."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from kayak.eval import _cli, _hint3, hint3, load_suite, prepare_hint3


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "hint3" / _hint3.REVISION
    directory.mkdir(parents=True)
    train = b"sentence,label\nA one,a\na   ONE,a\nA two,a\nA three,a\nB one,b\nB two,b\n"
    test = b"sentence,label\na ONE,a\nnew request,b\nunrelated,NO_NODES_DETECTED\n"
    pins = dict(_hint3.FILES)
    for name, raw in (
        ("LICENSE.md", b"fixture license"),
        ("sofmattress_train.csv", train),
        ("sofmattress_test.csv", test),
    ):
        (directory / name).write_bytes(raw)
        pins[name] = (len(raw), hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(_hint3, "FILES", pins)
    monkeypatch.setattr(_hint3, "COUNTS", {**_hint3.COUNTS, "sofmattress": (6, 3, 2)})
    return tmp_path


def test_grouped_development_uses_only_training_and_is_deterministic(cache: Path) -> None:
    (cache / "hint3" / _hint3.REVISION / "sofmattress_test.csv").unlink()
    train = hint3(cache_dir=cache, split="train")
    dev = hint3(cache_dir=cache)
    assert len(train.examples) + len(dev.examples) == 6
    train_keys = {_hint3._normalize(row.text) for row in train.examples}
    dev_keys = {_hint3._normalize(row.text) for row in dev.examples}
    assert not train_keys & dev_keys
    assert {row.label for row in train.examples} == {"a", "b"}
    assert {row.label for row in dev.examples} == {"a", "b"}
    assert hint3(cache_dir=cache).sha256 == dev.sha256
    assert dev.provenance["train_overlap_ids"] == "not inspected"
    assert list(dev.question.criteria) == ["a", "b", "NO_NODES_DETECTED"]


def test_official_test_order_overlap_and_explicit_sensitivity_split(cache: Path) -> None:
    official = hint3(cache_dir=cache, split="test")
    clean = hint3(cache_dir=cache, split="test", exclude_train_overlap=True)
    assert [row.id for row in official.examples] == [f"sofmattress:test:{i}" for i in range(3)]
    assert [row.id for row in clean.examples] == ["sofmattress:test:1", "sofmattress:test:2"]
    assert clean.split == "test-no-train-overlap"
    assert official.split == "test"
    assert official.sha256 != clean.sha256
    assert json.loads(official.provenance["train_overlap_ids"]) == ["sofmattress:test:0"]
    limited = hint3(cache_dir=cache, split="test", limit=1)
    assert limited.examples == official.examples[:1]
    assert limited.provenance["coverage"] == "subset"
    with pytest.raises(ValueError, match="only available for test"):
        hint3(cache_dir=cache, exclude_train_overlap=True)


@pytest.mark.parametrize("limit", [0, -1, True, 100])
def test_invalid_limits(cache: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        hint3(cache_dir=cache, limit=limit)


def test_integrity_and_missing_cache(cache: Path) -> None:
    path = cache / "hint3" / _hint3.REVISION / "sofmattress_train.csv"
    path.write_bytes(path.read_bytes() + b"tampering")
    with pytest.raises(ValueError, match="integrity"):
        hint3(cache_dir=cache)
    with pytest.raises(ValueError, match="integrity"):
        prepare_hint3(cache)
    path.unlink()
    with pytest.raises(ValueError, match="not cached"):
        hint3(cache_dir=cache)


def test_singletons_conflicts_and_annotation_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    from kayak.eval import Example

    rows = [Example(id="1", text="singleton", label="a")]
    assert _hint3._development_keys(rows) == set()
    rows.append(Example(id="2", text=" SINGLETON ", label="b"))
    with pytest.raises(ValueError, match="conflicting"):
        _hint3._development_keys(rows)
    monkeypatch.setattr(_hint3, "COUNTS", {"powerplay11": (1, 1, 1)})
    rows = _hint3._rows(b"sentence,label,\nmessage,a,private annotation\n", "powerplay11", "train")
    assert rows[0].text == "message"
    assert "annotation" not in rows[0].model_dump_json()
    for raw in (b"sentence,label\nmessage,a\n", b"sentence,label,\nmessage,a,note,extra\n"):
        with pytest.raises(ValueError):
            _hint3._rows(raw, "powerplay11", "train")


def test_prepare_is_explicit_bounded_and_train_only(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = cache / "hint3" / _hint3.REVISION
    raw = {path.name: path.read_bytes() for path in directory.iterdir()}
    for path in directory.iterdir():
        path.unlink()
    fetched = []

    @contextmanager
    def stream(method: str, url: str, **kwargs: object) -> Iterator[httpx.Response]:
        fetched.append(url.rsplit("/", 1)[1])
        yield httpx.Response(200, content=raw[fetched[-1]], request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "stream", stream)
    assert prepare_hint3(cache) == directory
    assert fetched == ["LICENSE.md", "sofmattress_train.csv"]
    prepare_hint3(cache)
    assert len(fetched) == 2
    prepare_hint3(cache, include_test=True)
    assert fetched[-1] == "sofmattress_test.csv"
    test = directory / "sofmattress_test.csv"
    test.unlink()
    raw[test.name] += b"oversized"
    with pytest.raises(ValueError, match="exceeds pinned size"):
        prepare_hint3(cache, include_test=True)
    assert not test.exists()
    raw[test.name] = b"x" * _hint3.FILES[test.name][0]
    with pytest.raises(ValueError, match="integrity"):
        prepare_hint3(cache, include_test=True)
    assert not test.exists()
    assert len(list(directory.iterdir())) == 2


def test_cli_suite_and_dataset_options(cache: Path) -> None:
    output = cache / "suite.json"
    args = ["suite", "hint3", "--data-cache-dir", str(cache), "--output", str(output)]
    assert _cli.main(args) == 0
    assert load_suite(output) == hint3(cache_dir=cache)
    assert _cli.main(args) == 2
    assert _cli.main(["prepare", "banking77", "--include-test"]) == 2
    assert (
        _cli.main(["run", "banking77", "--domain", "curekart", "--output", str(cache / "run")]) == 2
    )
    assert _cli.main(["prepare", "hint3", "--data-cache-dir", str(cache)]) == 0
