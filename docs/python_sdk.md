# Python SDK

This document defines the current Python SDK boundary for Kayak as it exists in
this monorepo.

## Status

What is verified:
- the public Python import surface is `import kayak`
- `python -m pip install .` from a source checkout works
- local editable consumption from Pixi works after adding Python
- the public API supports two explicit backends:
  - `numpy_reference`
  - `mojo_exact_cpu`

What is not claimed:
- a published `pip install kayak` workflow from PyPI
- a runtime without a local Mojo toolchain for `mojo_exact_cpu`
- a clean open-source split between the Python SDK and the proprietary engine

Those boundaries are based on verified commands run in this repo, not on an
intended future release shape.

## Supported Install Paths

Source checkout with pip:

```bash
python -m pip install .
```

Local editable consumption from another Pixi project:

```bash
pixi init .
pixi add python=3.11
pixi add --pypi --editable "kayak @ file:///absolute/path/to/kayak"
```

The second path is verified only for a local editable source checkout, not for a
published package index.

Important current boundary from fresh consumer-repo validation:
- the Pixi local-package path is verified for `numpy_reference`
- it is not yet verified for `mojo_exact_cpu`

The currently verified fresh-consumer path for `mojo_exact_cpu` is:

```bash
pixi init .
pixi add python=3.11 mojo
pixi run python -m ensurepip --default-pip
pixi run python -m pip install /absolute/path/to/kayak
```

That path works because the package build can bundle `kayak.mojopkg` when Mojo
is available during installation.

## Public Python API

Application code should import only from `kayak`.

Supported exports today:
- `LateQuery`
- `LateDocuments`
- `LateIndex`
- `LateScores`
- `SearchHit`
- `query`
- `documents`
- `packed_index`
- `hybrid_flat_dim128_index`
- `flat_query_dim128`
- `maxsim`
- `search`
- `NUMPY_REFERENCE_BACKEND`
- `MOJO_EXACT_CPU_BACKEND`

This contract is enforced by [python/tests/test_public_api_contract.py](/Users/teilomillet/Code/kayak-wt/python/tests/test_public_api_contract.py).

## Internal And Unstable Modules

The following are intentionally not the supported public SDK surface:
- `kayak_bridge`
- top-level Mojo package `kayak/`
- `benchmarks/`, `tests/`, and `examples/`
- storage artifact internals and benchmark-only modules

Reason:
- `kayak` is the stable Python entrypoint for application code
- `kayak_bridge` is currently an implementation layer inside the monorepo
- the top-level Mojo package is engine code, not the documented Python SDK

Imports from `kayak_bridge` should be treated as unstable.

## Quickstart

Reference backend quickstart:

```python
import numpy as np
import kayak

query = kayak.query(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
documents = kayak.documents(
    ["doc-a", "doc-b"],
    [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
    ],
)
index = documents.pack()

scores = kayak.maxsim(query, index, backend=kayak.NUMPY_REFERENCE_BACKEND)
hits = kayak.search(query, index, k=2, backend=kayak.NUMPY_REFERENCE_BACKEND)
```

Runnable example:
- [python/examples/quickstart.py](/Users/teilomillet/Code/kayak-wt/python/examples/quickstart.py)

Mojo exact CPU quickstart:

```python
scores = kayak.maxsim(query, index, backend=kayak.MOJO_EXACT_CPU_BACKEND)
```

Runnable example:
- [python/examples/mojo_exact_cpu_quickstart.py](/Users/teilomillet/Code/kayak-wt/python/examples/mojo_exact_cpu_quickstart.py)

## Backend Notes

`numpy_reference`:
- always the safest backend to rely on in docs and tests
- intended as the correctness-oriented reference path
- does not require Mojo

`mojo_exact_cpu`:
- uses a compiled Mojo extension module
- depends on the monorepo's Mojo engine package today
- is appropriate for internal use and controlled distribution
- should be treated as more operationally constrained than `numpy_reference`
- worked in fresh-consumer testing after `pip install /path/to/kayak` in an
  environment that already had `mojo`
- did not work in fresh-consumer testing after `pixi add --pypi "kayak @
  file:///..."` because that path did not produce a bundled `kayak.mojopkg`

## Recommendation

For the current monorepo phase:
- document `kayak` as the supported Python SDK
- keep `kayak_bridge` and the Mojo engine internal
- avoid promising a broader public stability boundary than the tests and
  packaging checks currently support
