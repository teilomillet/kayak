"""Pinned HINT3 v2 data; explicit downloads and duplicate-grouped development splits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Literal

import httpx

from ..decisions import Choice
from ._banking77 import DEFAULT_CACHE
from ._schema import Example, Suite

Domain = Literal["sofmattress", "curekart", "powerplay11"]
REVISION = "99965c892ea9a6801a87083d3861b0b04c91a4e7"
REPOSITORY = "https://github.com/hellohaptik/HINT3"
REJECT_LABEL = "NO_NODES_DETECTED"
COUNTS = {"sofmattress": (328, 397, 21), "curekart": (599, 991, 28), "powerplay11": (471, 983, 57)}
FILES = {
    "LICENSE.md": (281, "a5df081ff80f725f7bcada010abb9fc76ccb72afe7416c7a2665af022b300940"),
    "sofmattress_train.csv": (
        11866,
        "0addf4dc4b9df1e71509da5318d5a9f6e9a5d6eb98b9584d55cab3da7badf4b1",
    ),
    "sofmattress_test.csv": (
        19035,
        "9c4746f7cf445b5db276260e1c05b9e7b102fa2bca0ce1ace20f51df0ba53aae",
    ),
    "curekart_train.csv": (
        29653,
        "99620b357d20801cbbebe2823d606d1afecd9414074d97a7386aaf2493f8911f",
    ),
    "curekart_test.csv": (
        49506,
        "47bf651cafa8b8f476ca4bea3052ee9a29c5bc6a4d841f2162f12ad16064f5a0",
    ),
    "powerplay11_train.csv": (
        22670,
        "16847d06e62244ef902fe62a61261e8daec61d14335e535fcddf1752a8c8449b",
    ),
    "powerplay11_test.csv": (
        55838,
        "a2d5d03202e8ca116444162560647d0a7633c622fa782c09fe5cb7e196d462a9",
    ),
}


def _domain(domain: str) -> None:
    if domain not in COUNTS:
        raise ValueError("HINT3 domain must be sofmattress, curekart, or powerplay11")


def _verified(path: Path) -> bytes:
    size, expected = FILES[path.name]
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"HINT3 v2 integrity check failed: {path}")
    return raw


def prepare_hint3(
    cache_dir: str | Path = DEFAULT_CACHE,
    *,
    domain: Domain = "sofmattress",
    include_test: bool = False,
) -> Path:
    """Download one domain's pinned training data and license; test is opt-in."""
    _domain(domain)
    directory = Path(cache_dir) / "hint3" / REVISION
    directory.mkdir(parents=True, exist_ok=True)
    names = ["LICENSE.md", f"{domain}_train.csv"]
    if include_test:
        names.append(f"{domain}_test.csv")
    for name in names:
        path = directory / name
        if path.exists():
            _verified(path)
            continue
        split = "train" if name.endswith("_train.csv") else "test"
        remote = name if name == "LICENSE.md" else f"dataset/v2/{split}/{name}"
        size, expected = FILES[name]
        raw = bytearray()
        with httpx.stream(
            "GET",
            f"https://raw.githubusercontent.com/hellohaptik/HINT3/{REVISION}/{remote}",
            timeout=60,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > size:
                    raise ValueError(f"HINT3 download exceeds pinned size: {name}")
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"HINT3 download integrity check failed: {name}")
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


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _rows(raw: bytes, domain: Domain, split: str) -> list[Example]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = ["sentence", "label"]
    if domain == "powerplay11" and split == "train":
        columns.append("")  # Upstream annotation notes; never sent to inference.
    if reader.fieldnames != columns:
        raise ValueError("unexpected HINT3 CSV columns")
    rows = []
    for index, row in enumerate(reader):
        if set(row) != set(columns) or any(value is None for value in row.values()):
            raise ValueError("malformed HINT3 CSV row")
        rows.append(
            Example(id=f"{domain}:{split}:{index}", text=row["sentence"], label=row["label"])
        )
    if len(rows) != COUNTS[domain][0 if split == "train" else 1] or any(
        not r.text.strip() for r in rows
    ):
        raise ValueError("unexpected HINT3 row count or blank text")
    return rows


def _development_keys(rows: list[Example]) -> set[str]:
    labels: dict[str, str] = {}
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = _normalize(row.text)
        if key in labels and labels[key] != row.label:
            raise ValueError("conflicting HINT3 labels for normalized training text")
        labels[key] = row.label
        groups[row.label].add(key)
    selected = set()
    for keys in groups.values():
        count = max(1, len(keys) // 5) if len(keys) > 1 else 0
        ordered = sorted(
            keys, key=lambda key: hashlib.sha256(f"hint3-dev-v1:42:{key}".encode()).hexdigest()
        )
        selected.update(ordered[:count])
    return selected


def hint3(
    *,
    domain: Domain = "sofmattress",
    split: Literal["train", "dev", "test"] = "dev",
    cache_dir: str | Path = DEFAULT_CACHE,
    limit: int | None = None,
    exclude_train_overlap: bool = False,
) -> Suite:
    """Load offline; train/dev share no normalized text, and never read test files.

    Official test stays intact. The optional overlap-excluded variant declares
    a different split. Descriptions use training labels only. The reject option
    is a prediction, not guaranteed abstention. Limits select a source-order prefix.
    """
    _domain(domain)
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if exclude_train_overlap and split != "test":
        raise ValueError("overlap exclusion is only available for test")
    directory = Path(cache_dir) / "hint3" / REVISION
    try:
        _verified(directory / "LICENSE.md")
        training = _rows(_verified(directory / f"{domain}_train.csv"), domain, "train")
        rows = (
            _rows(_verified(directory / f"{domain}_test.csv"), domain, "test")
            if split == "test"
            else training
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "HINT3 is not cached; run 'kayak eval prepare hint3' "
            "with the domain and --include-test if needed"
        ) from exc
    labels = sorted({row.label for row in training})
    if len(labels) != COUNTS[domain][2] or REJECT_LABEL in labels:
        raise ValueError("unexpected HINT3 training label inventory")
    dev_keys = _development_keys(training)
    overlap = []
    if split == "test":
        training_keys = {_normalize(row.text) for row in training}
        overlap = [row.id for row in rows if _normalize(row.text) in training_keys]
        if exclude_train_overlap:
            excluded = set(overlap)
            rows = [row for row in rows if row.id not in excluded]
    else:
        rows = [row for row in rows if (_normalize(row.text) in dev_keys) == (split == "dev")]
    available = len(rows)
    if limit is not None and limit > available:
        raise ValueError(f"limit exceeds the {available} available examples")
    descriptions = {label: label.replace("_", " ").lower() for label in labels}
    descriptions[REJECT_LABEL] = "The request does not match any of the supported intents."
    source_split = "test" if split == "test" else "train"
    return Suite(
        name=f"hint3-v2-{domain}",
        split="test-no-train-overlap" if exclude_train_overlap else split,
        question=Choice(
            instructions=(
                "Which customer-support intent matches this request? "
                "Select NO_NODES_DETECTED if none of the supported intents match."
            ),
            criteria=descriptions,
        ),
        examples=rows[:limit],
        provenance={
            "repository": REPOSITORY,
            "revision": REVISION,
            "version": "v2",
            "source_split": source_split,
            "source_sha256": FILES[f"{domain}_{source_split}.csv"][1],
            "training_sha256": FILES[f"{domain}_train.csv"][1],
            "license": "ODbL-1.0; individual contents DbCL-1.0",
            "license_url": f"{REPOSITORY}/blob/{REVISION}/LICENSE.md",
            "citation": "Arora et al. (2020), https://aclanthology.org/2020.insights-1.16/",
            "partition": (
                "hint3-dev-v1:42; per-intent normalized groups; "
                "floor(20%), minimum one except singletons; training excludes dev"
            ),
            "normalization": "Unicode NFKC, casefold, whitespace collapse; punctuation retained",
            "selection": "original source order; limit selects a prefix, not a balanced sample",
            "candidate_recipe": (
                "hint3-label-text-v1; sorted training labels; "
                "underscores to spaces; explicit reject last"
            ),
            "reject_label": REJECT_LABEL,
            "annotation_notes": "unnamed upstream notes excluded from inference",
            "train_overlap_ids": json.dumps(overlap) if split == "test" else "not inspected",
            "overlap_excluded": str(exclude_train_overlap),
            "available_examples": str(available),
            "coverage": "full_split" if limit in (None, available) else "subset",
            "development_limit": (
                "training-derived dev has no OOS examples; insufficient to tune rejection"
            ),
        },
    )
