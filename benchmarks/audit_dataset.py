"""Inspect cached BANKING77 text overlap without downloading data or running a model."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Literal

from kayak.eval import Example, Suite, banking77

Normalization = Literal["exact", "casefold_whitespace", "unicode_word_tokens"]
NORMALIZATIONS: dict[Normalization, str] = {
    "exact": "text",
    "casefold_whitespace": '" ".join(text.casefold().split())',
    "unicode_word_tokens": '" ".join(re.findall(r"\\w+", text.casefold()))',
}


def normalize(text: str, normalization: Normalization) -> str:
    if normalization == "exact":
        return text
    if normalization == "casefold_whitespace":
        return " ".join(text.casefold().split())
    return " ".join(re.findall(r"\w+", text.casefold()))


def records(examples: list[Example]) -> list[dict[str, str]]:
    return [
        {"id": example.id, "text": example.text, "label": example.label} for example in examples
    ]


def audit_suites(suites: list[Suite]) -> dict[str, object]:
    """Keep every row; report diagnostic groups and their row/pair counts separately."""
    if not suites or len({suite.split for suite in suites}) != len(suites):
        raise ValueError("suites must be nonempty and have distinct split names")

    manifests: dict[str, object] = {}
    for suite in suites:
        manifests[suite.split] = {
            "name": suite.name,
            "split": suite.split,
            "suite_sha256": suite.sha256,
            "example_count": len(suite.examples),
            "provenance": dict(suite.provenance),
        }

    analyses: dict[str, object] = {}
    for normalization, definition in NORMALIZATIONS.items():
        by_split: dict[str, dict[str, list[Example]]] = {}
        within: dict[str, object] = {}
        for suite in suites:
            grouped: dict[str, list[Example]] = defaultdict(list)
            for example in sorted(suite.examples, key=lambda row: row.id):
                grouped[normalize(example.text, normalization)].append(example)
            by_split[suite.split] = grouped

            duplicates: list[dict[str, object]] = []
            duplicate_rows = duplicate_pairs = conflicting_groups = 0
            for key in sorted(grouped):
                examples = grouped[key]
                if len(examples) < 2:
                    continue
                conflicting = len({example.label for example in examples}) > 1
                duplicates.append(
                    {
                        "normalized_text": key,
                        "rows": records(examples),
                        "conflicting_labels": conflicting,
                    }
                )
                duplicate_rows += len(examples)
                duplicate_pairs += len(examples) * (len(examples) - 1) // 2
                conflicting_groups += conflicting
            within[suite.split] = {
                "row_count": len(suite.examples),
                "distinct_key_count": len(grouped),
                "duplicate_group_count": len(duplicates),
                "duplicate_row_count": duplicate_rows,
                "duplicate_pair_count": duplicate_pairs,
                "conflicting_label_group_count": conflicting_groups,
                "groups": duplicates,
            }

        between: list[dict[str, object]] = []
        for index, left_suite in enumerate(suites):
            for right_suite in suites[index + 1 :]:
                left = by_split[left_suite.split]
                right = by_split[right_suite.split]
                shared_keys = sorted(left.keys() & right.keys())
                overlaps: list[dict[str, object]] = []
                left_rows = right_rows = row_pairs = conflicting_groups = 0
                for key in shared_keys:
                    labels = {example.label for example in [*left[key], *right[key]]}
                    conflicting = len(labels) > 1
                    overlaps.append(
                        {
                            "normalized_text": key,
                            "left_rows": records(left[key]),
                            "right_rows": records(right[key]),
                            "conflicting_labels": conflicting,
                        }
                    )
                    left_rows += len(left[key])
                    right_rows += len(right[key])
                    row_pairs += len(left[key]) * len(right[key])
                    conflicting_groups += conflicting
                between.append(
                    {
                        "left_split": left_suite.split,
                        "right_split": right_suite.split,
                        "overlap_group_count": len(overlaps),
                        "left_row_count": left_rows,
                        "right_row_count": right_rows,
                        "row_pair_count": row_pairs,
                        "conflicting_label_group_count": conflicting_groups,
                        "groups": overlaps,
                    }
                )
        analyses[normalization] = {"definition": definition, "within": within, "between": between}

    return {
        "schema_version": 1,
        "unicode_version": unicodedata.unidata_version,
        "interpretation": (
            "Diagnostic text overlap only; normalization does not establish semantic "
            "equivalence or model-training contamination. No rows or labels are changed."
        ),
        "count_definitions": {
            "group": "One shared normalized-text key; groups are sorted by that key.",
            "duplicate_row_count": "All rows in within-split groups containing at least two rows.",
            "duplicate_pair_count": "Unordered within-split row pairs: sum(n * (n - 1) / 2).",
            "left_row_count": "Left-split rows in cross-split overlap groups.",
            "right_row_count": "Right-split rows in cross-split overlap groups.",
            "row_pair_count": "Cross-split row pairs: sum(left rows * right rows) per shared key.",
            "conflicting_label_group_count": "Groups containing more than one distinct gold label.",
            "row_order": "Rows within each group are sorted lexicographically by original ID.",
        },
        "suites": manifests,
        "analyses": analyses,
    }


class Arguments(argparse.Namespace):
    data_cache_dir: Path | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache-dir", type=Path, help="existing prepared dataset cache")
    args = parser.parse_args(argv, namespace=Arguments())
    splits: tuple[Literal["train", "dev", "test"], ...] = ("train", "dev", "test")
    suites: list[Suite] = []
    try:
        for split in splits:
            if args.data_cache_dir is None:
                suite = banking77(split=split)
            else:
                suite = banking77(split=split, cache_dir=args.data_cache_dir)
            suites.append(suite)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(audit_suites(suites), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
