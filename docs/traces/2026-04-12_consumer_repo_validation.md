# Consumer Repo Validation Trace

Date: `2026-04-12`

## Scope

This trace records validation of the Python SDK from fresh repositories outside
the Kayak checkout.

The goal was to test actual consumer workflows rather than rely only on
in-repo smoke checks.

## Validation Cases

### Case 1: Fresh Pixi repo with `python` plus local Kayak package

Setup:

```bash
pixi init .
pixi add python=3.11
pixi add --pypi "kayak @ file:///absolute/path/to/kayak"
```

Verified:
- `import kayak` works
- `numpy_reference` scoring and search work

Observed failure:
- `mojo_exact_cpu` raised:
  `RuntimeError: Kayak could not find Mojo sources or a bundled kayak.mojopkg artifact.`

Interpretation:
- this path is currently good for the pure Python SDK path
- it is not enough for the Mojo backend

### Case 2: Fresh Pixi repo with `python`, `mojo`, and local Kayak package via `pixi add`

Setup:

```bash
pixi init .
pixi add python=3.11 mojo
pixi add --pypi "kayak @ file:///absolute/path/to/kayak"
```

Verified:
- `numpy_reference` still works

Observed failure:
- `mojo_exact_cpu` still failed with the same missing `kayak.mojopkg` error

Interpretation:
- having `mojo` in the consumer environment is not sufficient by itself
- this install path did not bundle the Mojo package artifact during installation

### Case 3: Fresh Pixi repo with `python`, `mojo`, and Kayak installed through `pip`

Setup:

```bash
pixi init .
pixi add python=3.11 mojo
pixi run python -m ensurepip --default-pip
pixi run python -m pip install /absolute/path/to/kayak
```

Verified:
- `import kayak` works
- `numpy_reference` works
- `mojo_exact_cpu` works
- bundled artifact exists at:
  `.../site-packages/kayak_bridge/_artifacts/kayak.mojopkg`

## Result

What the evidence supports:
- the Python SDK works from a fresh external repo
- `numpy_reference` has the least constrained install path
- `mojo_exact_cpu` works when installation happens in an environment where Mojo
  is available and the build bundles `kayak.mojopkg`

What the evidence does not support:
- claiming that every supported install path also supports `mojo_exact_cpu`
- claiming that the local Pixi package-add flow is sufficient for the Mojo
  backend

Current conclusion:
- document `numpy_reference` as the safest general-purpose backend
- document `mojo_exact_cpu` as a more constrained internal/distribution path
- avoid overstating the equivalence of `pip install` and `pixi add` for the
  Mojo backend
