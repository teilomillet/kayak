# Python Mojo Exact CPU Trace

Date: `2026-04-12`

## Scope

This trace records the step where the Python late-interaction layer gained a
real Mojo-backed exact scoring backend and a pip-installable package layout from
the existing source tree.

What changed:
- `mojo_exact_cpu` backend behind the existing Python `backend=` seam
- explicit Mojo Python bindings in
  `python/kayak_bridge/_mojo_exact_cpu_bindings.mojo`
- on-demand build and load of a Mojo extension module from Python
- setuptools packaging rooted at `python/`, with bundled `kayak.mojopkg` when
  Mojo is available at build time

## Why This Shape

This keeps the public Python API additive and stable:
- Python still owns the ergonomic object model
- Mojo still owns the authoritative exact scoring kernels
- backend choice stays explicit instead of hidden behind magic dispatch

Keeping `python/` as the package root is deliberate.
The repo already has a top-level `kayak/` Mojo package, so adding a second
`src/kayak` tree would make the package layout harder to browse and reason
about. Pointing setuptools at `python/` preserves a single Python package tree
without disturbing the Mojo source tree.

## Validation

Commands:

```bash
pixi run test_python_api
pixi run test_python_bridge
pixi run python -m pip install . --target /tmp/<target>
```

Verified:
- `numpy_reference` and `mojo_exact_cpu` agree on packed exact scores
- `numpy_reference` and `mojo_exact_cpu` agree on flat-query plus hybrid-flat
  exact scores
- `pip install .` into an isolated target directory exposes `import kayak`
- the installed package can execute `mojo_exact_cpu` scoring through the bundled
  `kayak.mojopkg`

## Boundary

What this evidence supports:
- Kayak's Python API is now packageable from the existing repo
- the Python facade can call real Mojo exact kernels without changing the public
  object model
- the backend boundary is explicit and test-covered

What this evidence does not support:
- that a published wheel can run without a local Mojo toolchain
- that all future backends should follow the same binding strategy

Current conclusion:
- keep `numpy_reference` as the always-available reference path
- keep `mojo_exact_cpu` as the explicit fast exact backend
- use the same Python object model as the stable contract for future backends
