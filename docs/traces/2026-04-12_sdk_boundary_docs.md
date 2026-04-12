# SDK Boundary And Docs Trace

Date: `2026-04-12`

## Scope

This trace records the step where the repo's Python SDK boundary was made
explicit without attempting a premature open-source split.

What changed:
- explicit public API contract under `python/kayak/__init__.py`
- explicit internal/unstable marking for `kayak_bridge`
- runnable Python quickstart examples
- SDK install and usage documentation under `docs/python_sdk.md`
- public API contract tests

## Why This Shape

The current monorepo still has direct coupling between the public Python package
and the Mojo engine through the `mojo_exact_cpu` backend. That means a clean SDK
vs proprietary-engine split is not yet verified.

The least risky step is therefore:
- keep the monorepo
- document the supported public boundary narrowly
- mark internal modules plainly
- verify the actual install and usage paths we claim

## Validation

Commands:

```bash
pixi run test_python_api
pixi run test_python_bridge
pixi run demo_python_sdk
pixi run demo_python_sdk_mojo
```

Verified:
- the public Python API contract is exercised in tests
- direct use of `import kayak` works for the documented quickstarts
- the reference and Mojo exact backends both execute through the public package

## Boundary

Supported public surface:
- `import kayak`
- the names exported from `kayak.PUBLIC_API`

Explicitly internal or unstable:
- `kayak_bridge`
- top-level Mojo package `kayak/`
- benchmark, test, and storage implementation details

Current conclusion:
- document the Python SDK as a narrow public surface
- keep the rest of the repo internal while the product boundary is still moving
