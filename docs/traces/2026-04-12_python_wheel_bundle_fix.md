# Python Wheel Bundle Fix

Date: `2026-04-12`

## Claim

Repo-head Python distributions should carry enough Mojo engine material to make
the `mojo_exact_cpu` backend usable after install.

Concretely:
- the source distribution should include the top-level `kayak/` Mojo sources so
  wheel builds from sdist can still stage engine sources
- the wheel should include bundled engine sources under
  `kayak_bridge/_engine/kayak/...`
- when Mojo is available at build time, the wheel should also include
  `kayak_bridge/_artifacts/kayak.mojopkg`
- a fresh consumer install from that wheel should run `mojo_exact_cpu` when the
  consumer environment has `mojo`

## Design Choice

Implemented:
- `setup.py` now stages the top-level `kayak/` Mojo sources into
  `kayak_bridge/_engine/kayak` inside the built Python package
- `setup.py` now prefers `pixi run mojo` over the raw `.pixi/.../mojo` binary
  because the raw binary failed to resolve Mojo's `std` modules during
  packaging on the validation machine
- `python/kayak_bridge/mojo_exact_cpu.py` now falls back to bundled engine
  sources when a bundled `kayak.mojopkg` artifact is absent
- `MANIFEST.in` now includes the top-level `kayak/**/*.mojo` sources in the
  sdist

Why this direction:
- bundling sources covers builds that happen without Mojo
- bundling `kayak.mojopkg` covers the faster path when Mojo is available during
  build
- carrying both is additive and keeps the backend explicit rather than magical

## Evidence

### Source-level regression checks

Command:
- `pixi run test_python_api`

Result:
- passed, `12` tests

Added focused packaging checks:
- `python/tests/test_mojo_packaging.py`

These cover:
- fallback to bundled engine sources
- preference for a bundled `kayak.mojopkg` artifact when present

### Distribution build

Command:
- `uv build --out-dir /tmp/kayak-dist-test`

Observed:
- build succeeded
- produced:
  - `/tmp/kayak-dist-test-v012/kayak-0.1.2.tar.gz`
  - `/tmp/kayak-dist-test-v012/kayak-0.1.2-py3-none-any.whl`

Wheel contents included:
- `kayak_bridge/_artifacts/kayak.mojopkg`
- `kayak_bridge/_engine/kayak/__init__.mojo`
- the rest of the bundled Mojo engine tree

The sdist build log also showed the top-level `kayak/` Mojo sources being copied
into the source distribution.

### Fresh consumer install from wheel

Environment:
- fresh Pixi project
- `python=3.11`
- `mojo`

Install path:
- `pixi run python -m ensurepip --upgrade`
- `pixi run python -m pip install /tmp/kayak-dist-test-v012/kayak-0.1.2-py3-none-any.whl`

Observed after install:
- `kayak_bridge/_artifacts/kayak.mojopkg` existed
- `kayak_bridge/_engine/kayak/__init__.mojo` existed
- `numpy_reference` scoring worked
- `mojo_exact_cpu` scoring worked

Minimal output:
- `numpy_scores [2.0, 1.0]`
- `mojo_scores [2.0, 1.0]`
- `mojo_hits [('doc-a', 2.0), ('doc-b', 1.0)]`

## Conclusion

Repo-head packaging now carries the Mojo backend resources needed for a fresh
consumer wheel install to execute `mojo_exact_cpu`, provided the consumer
environment has a working Mojo CLI.

Important boundary:
- the already-published `kayak 0.1.1` package predates this fix and was
  separately verified to be missing the bundled artifact
