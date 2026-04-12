# Python Late-Interaction Layer Trace

Date: `2026-04-12`

## Scope

This trace records the first additive Python-facing object layer for late
interaction.

What changed:
- explicit Python `LateQuery`, `LateDocuments`, `LateIndex`, and `LateScores`
  objects under `python/kayak_bridge/`
- explicit query and index layout conversions for `flat_dim128` and
  `hybrid_flat_dim128`
- a reference exact `maxsim` and `search` backend exposed as
  `numpy_reference`
- a light `python/kayak/` facade so the API can read as `kayak.query(...)`
  under `PYTHONPATH=python`

## Why This Shape

This is additive rather than a rewrite because the verified repo shape already
separates:
- late-interaction contracts
- layout transforms
- scoring kernels
- search orchestration

The Python layer mirrors that separation instead of flattening late interaction
into fake dense tensors. Query vector count, document vector count, and layout
remain first-class metadata.

## Validation

Command:

```bash
PYTHONPATH=python pixi run python -m unittest discover -s python/tests -p 'test_*.py'
```

Passed.

Verified:
- PyTorch query input converts into explicit late-interaction query objects
- packed index construction preserves document offsets and vector counts
- `flat_dim128` conversion is rejected for non-`128`-dim queries
- exact scores match across packed and hybrid-flat layouts
- top-k search preserves stable tie order
- scores roundtrip back to PyTorch tensors

## Boundary

What this evidence supports:
- Kayak now has a domain-native Python object model for late interaction
- the public API can be ergonomic without hiding the ragged structure
- layout-specific conversions and exact scoring are testable today

What this evidence does not support:
- claiming that Python calls are already backed by the Mojo exact-search kernels
- claiming that the NumPy reference path is a serving-grade performance path

Current conclusion:
- keep the Python object layer explicit and exact
- use `numpy_reference` as the correctness-oriented backend today
- wire a future Python-to-Mojo backend through the same object model rather than
  creating a second public API
