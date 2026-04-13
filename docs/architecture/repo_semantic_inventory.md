# Repo Semantic Inventory

This document owns the repo-wide semantic inventory guardrail.

It does not own:
- snapshot artifact inventory inside sealed collections
- per-query search-plan selection
- benchmark results themselves

Those are separate runtime concerns.

## Problem

The repository already had many good social signals:
- narrow folders
- explicit tests
- strong trace notes
- architecture docs with real contracts

That was useful, but it still left one structural gap:

- a new tracked file could land without any explicit semantic classification
- an old ownership rule could silently become stale
- traces and architecture notes could drift without a single canonical map of
  what each tracked part of the repo is for

That is semantic drift.

## Guardrail

The root file [`repo_semantic_inventory.toml`](../../repo_semantic_inventory.toml)
is now the canonical repo-level semantic map.

It enforces three invariants:

1. Every tracked file must match exactly one inventory rule.
2. Source and entrypoint rules must declare the normative docs and tests that
   guard their meaning.
3. Inventory rules are not allowed to go stale; a rule that matches no tracked
   files is a test failure.

Reason:
- uncovered files are unowned semantics
- multiply matched files mean the ownership model is ambiguous
- stale rules mean the ownership model no longer reflects the tree

The guardrail intentionally walks `git ls-files`, not every file in the working
tree.

Reason:
- repo semantics should attach to versioned surfaces
- local caches, scratch notes, and temporary outputs would otherwise create
  noisy false positives
- concurrent untracked work becomes enforceable as soon as it is tracked

## Role Meanings

- `normative_doc`: stable contract-setting documentation
- `design_doc`: strategy, rationale, or roadmap documentation
- `evidence_doc`: point-in-time trace notes and implementation evidence
- `engine_source`: Mojo engine modules
- `benchmark_support_source`: reusable benchmark-support code
- `benchmark_entrypoint`: runnable benchmark entrypoints
- `python_sdk_source`: public or internal Python SDK implementation
- `example_surface`: runnable examples meant for users or developers
- `utility_script`: repo maintenance or dataset-prep scripts
- `packaging_control`: build, packaging, and task-entrypoint control files
- `repo_control`: root policy and project-map files
- `test_suite`: verification files
- `lockfile` and `legal_notice`: tracked metadata that should stay explicit but
  are not semantic engine modules

## Why This Is Separate From Snapshot Inventory

The runtime snapshot inventory answers:
- which sealed artifacts exist for a collection snapshot?

The repo semantic inventory answers:
- what is each tracked file in this repository allowed to mean?

They solve different drift problems.

The snapshot inventory protects runtime planner correctness.
The repo semantic inventory protects codebase comprehension and merge safety.

## Verification

Run:

```bash
PYTHONPATH=python python -m unittest python/tests/test_repo_semantic_inventory.py
```

That test walks `git ls-files`, checks every tracked path against the inventory,
and fails on:
- uncovered files
- multiply classified files
- stale rules
- missing required doc/test links for source and entrypoint rules
