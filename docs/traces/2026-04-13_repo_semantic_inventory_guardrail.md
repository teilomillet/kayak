# 2026-04-13: repo semantic inventory guardrail

## Claim

Semantic drift is now mechanically harder at the repository level.

Reason:
- every tracked file must now belong to exactly one explicit semantic bucket
- source and entrypoint buckets must point at live docs and tests
- stale ownership buckets fail instead of lingering silently

## Files Added Or Updated

- `repo_semantic_inventory.toml`
- `docs/architecture/repo_semantic_inventory.md`
- `python/tests/test_repo_semantic_inventory.py`
- `README.md`
- `pyproject.toml`

## What The Guardrail Checks

The new Python test walks `git ls-files` and fails on:
- uncovered tracked files
- multiply classified tracked files
- stale inventory rules that match no tracked files
- source or entrypoint rules that do not point at live docs or tests

That is a different guarantee from snapshot artifact inventory.

Snapshot inventory answers:
- which artifacts exist for a sealed collection snapshot?

Repo semantic inventory answers:
- what is each tracked file in this tree allowed to mean?

The test intentionally scopes itself to tracked files rather than every file in
the working tree.

Reason:
- caches and scratch files are not stable repo semantics
- concurrent untracked work should not create false positives
- once a file becomes tracked, the inventory immediately starts enforcing it

## Inventory Summary

At verification time the inventory classified the entire tracked tree into
explicit buckets, including:

- repo policy and packaging control files at the root
- benchmark entrypoints under `benchmarks/`
- normative architecture docs under `docs/architecture/`
- evidence notes under `docs/traces/`
- engine source folders under `kayak/`
- Python SDK and bridge code under `python/`
- Mojo and Python test suites

The intent is not to claim that one test proves semantic correctness.

The narrower claim is:
- the repository now has a canonical machine-readable ownership map
- additions and moves can no longer bypass that map silently

## Verification

Commands run:

```bash
pixi run test_repo_semantic_inventory
pixi run test_python_api
```

Observed results:

- `pixi run test_repo_semantic_inventory`
  - passed
  - validated exact-one-rule coverage for the full tracked tree
- `pixi run test_python_api`
  - passed
  - `45` Python tests

## Notes

The first inventory implementation failed for a good reason:
- loose glob matching let `docs/*.md` overlap with `docs/architecture/*.md`
  and `docs/traces/*.md`

That failure was kept as evidence that the test is doing real work.

The final matcher is path-segment anchored, so a rule only covers the path
family it explicitly declares.
