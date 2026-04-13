from __future__ import annotations

import fnmatch
import subprocess
import tomllib
import unittest
from dataclasses import dataclass

from kayak_bridge.cache_paths import REPO_ROOT


INVENTORY_PATH = REPO_ROOT / "repo_semantic_inventory.toml"

KNOWN_ROLES = {
    "benchmark_entrypoint",
    "benchmark_support_source",
    "design_doc",
    "engine_source",
    "evidence_doc",
    "example_surface",
    "legal_notice",
    "lockfile",
    "normative_doc",
    "packaging_control",
    "python_sdk_source",
    "repo_control",
    "test_suite",
    "utility_script",
}

ROLES_REQUIRING_DOCS = {
    "benchmark_entrypoint",
    "benchmark_support_source",
    "engine_source",
    "example_surface",
    "packaging_control",
    "python_sdk_source",
    "repo_control",
    "utility_script",
}

ROLES_REQUIRING_TESTS = {
    "benchmark_entrypoint",
    "benchmark_support_source",
    "engine_source",
    "packaging_control",
    "python_sdk_source",
}


@dataclass(frozen=True)
class InventoryRule:
    name: str
    role: str
    why: str
    paths: tuple[str, ...]
    normative_docs: tuple[str, ...]
    tests: tuple[str, ...]


def _tracked_files() -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
    )
    paths = tuple(line for line in output.splitlines() if line)
    if not paths:
        raise AssertionError("git ls-files returned no tracked files")
    return paths


def _load_inventory_rules() -> tuple[InventoryRule, ...]:
    data = tomllib.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise AssertionError("repo semantic inventory version must be 1")
    raw_rules = data.get("rule")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise AssertionError("repo semantic inventory must define at least one rule")

    rules: list[InventoryRule] = []
    seen_names: set[str] = set()
    for entry in raw_rules:
        name = _require_non_empty_string(entry.get("name"), "rule name")
        role = _require_non_empty_string(entry.get("role"), f"{name} role")
        why = _require_non_empty_string(entry.get("why"), f"{name} why")
        if role not in KNOWN_ROLES:
            raise AssertionError(f"{name} uses unknown role {role!r}")
        if name in seen_names:
            raise AssertionError(f"duplicate inventory rule name: {name}")
        seen_names.add(name)
        rules.append(
            InventoryRule(
                name=name,
                role=role,
                why=why,
                paths=_require_string_list(entry.get("paths"), f"{name} paths"),
                normative_docs=_optional_string_list(entry.get("normative_docs")),
                tests=_optional_string_list(entry.get("tests")),
            )
        )
    return tuple(rules)


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssertionError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    return _require_string_list(value, "optional string list")


def _require_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AssertionError(f"{field_name} must be a non-empty list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AssertionError(f"{field_name} must contain only non-empty strings")
        items.append(item.strip())
    if len(set(items)) != len(items):
        raise AssertionError(f"{field_name} must not contain duplicates")
    return tuple(items)


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(_path_matches(path, pattern) for pattern in patterns)


def _path_matches(path: str, pattern: str) -> bool:
    path_parts = path.split("/")
    pattern_parts = pattern.split("/")
    if len(path_parts) != len(pattern_parts):
        return False
    return all(
        fnmatch.fnmatchcase(path_part, pattern_part)
        for path_part, pattern_part in zip(path_parts, pattern_parts)
    )


def _matching_paths(paths: tuple[str, ...], patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if _matches_any(path, patterns))


class RepoSemanticInventoryTests(unittest.TestCase):
    def test_every_tracked_file_matches_exactly_one_inventory_rule(self) -> None:
        tracked_files = _tracked_files()
        rules = _load_inventory_rules()

        unmatched: list[str] = []
        ambiguous: list[tuple[str, tuple[str, ...]]] = []
        for path in tracked_files:
            matching_rules = tuple(rule.name for rule in rules if _matches_any(path, rule.paths))
            if not matching_rules:
                unmatched.append(path)
            elif len(matching_rules) > 1:
                ambiguous.append((path, matching_rules))

        self.assertEqual(unmatched, [], f"uncovered tracked files: {unmatched}")
        self.assertEqual(
            ambiguous,
            [],
            "multiply classified tracked files: "
            + ", ".join(f"{path} -> {names}" for path, names in ambiguous),
        )

    def test_inventory_rules_are_live_and_have_required_guardrails(self) -> None:
        tracked_files = _tracked_files()
        rules = _load_inventory_rules()

        for rule in rules:
            matched_paths = _matching_paths(tracked_files, rule.paths)
            self.assertNotEqual(
                matched_paths,
                (),
                f"inventory rule {rule.name!r} is stale and matches no tracked files",
            )
            if rule.role in ROLES_REQUIRING_DOCS:
                self.assertNotEqual(
                    rule.normative_docs,
                    (),
                    f"{rule.name!r} must declare normative_docs",
                )
                matched_docs = _matching_paths(tracked_files, rule.normative_docs)
                self.assertNotEqual(
                    matched_docs,
                    (),
                    f"{rule.name!r} must point at live normative docs",
                )
            if rule.role in ROLES_REQUIRING_TESTS:
                self.assertNotEqual(rule.tests, (), f"{rule.name!r} must declare tests")
                matched_tests = _matching_paths(tracked_files, rule.tests)
                self.assertNotEqual(
                    matched_tests,
                    (),
                    f"{rule.name!r} must point at live tests",
                )

    def test_architecture_doc_records_repo_semantic_inventory_contract(self) -> None:
        contract_doc = (
            REPO_ROOT / "docs" / "architecture" / "repo_semantic_inventory.md"
        ).read_text(encoding="utf-8")

        self.assertIn("repo_semantic_inventory.toml", contract_doc)
        self.assertIn("Every tracked file must match exactly one inventory rule.", contract_doc)
        self.assertIn("point-in-time trace notes", contract_doc)


if __name__ == "__main__":
    unittest.main()
