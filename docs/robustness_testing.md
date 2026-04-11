# Robustness Testing

This repository borrows testing discipline from:

- Ordeal's emphasis on properties, fault-oriented exploration, and mutation as a test-quality check.
- Antithesis's emphasis on stating properties directly and testing invariant violations instead of only enumerating examples.

The current implementation is intentionally narrower than those systems.
`kayak` is a Mojo-first retrieval core, so the immediate goal is not full autonomous chaos infrastructure.
The goal is a sound, cheap, deterministic robustness layer that fits this codebase today.

## What Exists Now

- `tests/test_battle.mojo`
  - deterministic randomized differential checks against an independent reference scorer
  - metamorphic invariants for document order, zero-vector appends, and positive query scaling
  - edge-case checks for top-k behavior and index construction
- `tests/test_storage_invariants.mojo`
  - storage compatibility checks
  - storage corruption checks
- `tests/test_eval_battle.mojo`
  - metric reference checks
  - monotonicity and empty-task guard checks
- `python/scripts/mutation_smoke.py`
  - a curated mutation-smoke harness against retrieval, evaluation, and storage invariants

## Why This Shape

This shape is justified by the current system boundary:

- the retrieval core is deterministic and CPU-first
- the strongest current risks are arithmetic regressions, ordering bugs, shape-validation mistakes, and storage-compatibility mistakes
- these are well served by differential tests, metamorphic tests, and curated mutation checks

## What This Does Not Claim

- It is not full mutation testing coverage.
- It is not deterministic multiverse or fault-scheduling infrastructure.
- It does not replace benchmark validation on public retrieval datasets.

## Commands

```bash
pixi run test_battle
pixi run test_storage_invariants
pixi run test_eval_battle
pixi run mutate_smoke
```

## Next Good Expansions

- add corruption tests for more stored payload files
- add workload-size sweeps to the battle tests
- add future backend differential tests once a GPU path exists
- add a more systematic mutant catalog as the codebase grows
