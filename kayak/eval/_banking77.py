"""Pinned, integrity-checked BANKING77 data and a fixed development partition."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Literal

import httpx

from ..decisions import Choice
from ._schema import Example, Suite

REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
REPOSITORY = "https://github.com/PolyAI-LDN/task-specific-datasets"
DEFAULT_CACHE = Path(".benchmarks/datasets")
FILES = {
    "categories.json": (2036, "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63"),
    "train.csv": (839073, "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "test.csv": (239961, "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
    "LICENSE": (18650, "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661"),
}
INSTRUCTIONS = "Which banking intent best matches this customer request?"


def _verified(path: Path) -> bytes:
    size, expected = FILES[path.name]
    # Bound even a corrupt local cache before reading it into memory.
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"BANKING77 integrity check failed: {path}")
    return raw


def prepare_banking77(cache_dir: str | Path = DEFAULT_CACHE) -> Path:
    """Explicitly download missing official files; never replace a corrupt cache."""
    directory = Path(cache_dir) / "banking77" / REVISION
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in FILES.items():
        path = directory / name
        if path.exists():
            _verified(path)
            continue
        remote = name if name == "LICENSE" else f"banking_data/{name}"
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/{remote}"
        raw = bytearray()
        with httpx.stream("GET", url, timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > size:
                    raise ValueError(f"BANKING77 download exceeds pinned size: {name}")
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"BANKING77 download integrity check failed: {name}")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return directory


def _rows(raw: bytes, split: str, labels: list[str]) -> list[Example]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if reader.fieldnames != ["text", "category"]:
        raise ValueError("unexpected BANKING77 CSV columns")
    rows = [
        Example(id=f"{split}:{i}", text=row["text"], label=row["category"])
        for i, row in enumerate(reader)
    ]
    expected_count = 10003 if split == "train" else 3080
    if len(rows) != expected_count or {row.label for row in rows} != set(labels):
        raise ValueError("unexpected BANKING77 row count or category set")
    return rows


def _selection_key(example: Example) -> str:
    return hashlib.sha256(f"banking77-partition-v1:42:{example.id}".encode()).hexdigest()


def banking77(
    *,
    split: Literal["train", "dev", "test"] = "dev",
    cache_dir: str | Path = DEFAULT_CACHE,
    limit: int | None = None,
) -> Suite:
    """Load cached data, with 10 train examples/intent reserved for development.

    All 77 candidates remain present even for a limited smoke run. Selection is
    deterministic, round-robin across intents; execution shuffling is separate.
    No data is downloaded implicitly, and the official test split is unchanged.
    """
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    directory = Path(cache_dir) / "banking77" / REVISION
    try:
        categories: object = json.loads(_verified(directory / "categories.json"))
        _verified(directory / "LICENSE")
        source_split = "test" if split == "test" else "train"
        raw = _verified(directory / f"{source_split}.csv")
    except FileNotFoundError as exc:
        raise ValueError(
            "BANKING77 is not cached; run 'kayak eval prepare banking77' first"
        ) from exc
    if (
        not isinstance(categories, list)
        or len(categories) != 77
        or not all(isinstance(label, str) and label.strip() for label in categories)
        or len(set(categories)) != 77
    ):
        raise ValueError("BANKING77 must define exactly 77 distinct named categories")
    labels = [str(label) for label in categories]
    rows = _rows(raw, source_split, labels)
    grouped: dict[str, list[Example]] = defaultdict(list)
    for row in rows:
        grouped[row.label].append(row)
    for label in labels:
        grouped[label].sort(key=_selection_key)
        if split == "dev":
            grouped[label] = grouped[label][:10]
        elif split == "train":
            grouped[label] = grouped[label][10:]
        elif len(grouped[label]) != 40:
            raise ValueError("BANKING77 test must have 40 examples per intent")
    selected: list[Example] = []
    largest_class = max(len(examples) for examples in grouped.values())
    for position in range(largest_class):
        for label in labels:
            if position < len(grouped[label]):
                selected.append(grouped[label][position])
    available = len(selected)
    if limit is not None and limit > available:
        raise ValueError(f"limit exceeds the {available} available {split} examples")
    return Suite(
        name="banking77",
        split=split,
        question=Choice(
            instructions=INSTRUCTIONS,
            criteria={label: label.replace("_", " ") for label in labels},
        ),
        examples=selected[:limit],
        provenance={
            "repository": REPOSITORY,
            "revision": REVISION,
            "source_split": source_split,
            "source_sha256": FILES[f"{source_split}.csv"][1],
            "categories_sha256": FILES["categories.json"][1],
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Casanueva et al. (2020), https://aclanthology.org/2020.nlp4convai-1.5/",
            "partition": "sha256-v1:42; 10/intent from train for dev; train excludes dev",
            "candidate_recipe": "banking77-label-text-v1; underscores replaced by spaces",
            "selection": "round-robin intents, SHA-256 order within each intent",
            "available_examples": str(available),
            "coverage": "full_split" if limit in (None, available) else "subset",
            "task": "zero-shot 77-way label-description classification; no demonstrations",
        },
    )
