# Published Package Validation

Date: `2026-04-12`

## Claim

The newly published `kayak` package should be usable from fresh consumer
projects through the install flows users are most likely to try:
- `pip install kayak`
- `uv add kayak`
- `pixi add --pypi kayak`

For the Mojo-backed path, the relevant claim is stronger:
- the published package should also support `mojo_exact_cpu` in a fresh
  consumer environment

## Evidence Needed

- fresh environments, not the repo checkout
- the installed package version
- a minimal `numpy_reference` scoring call
- whether `kayak_bridge/_artifacts/kayak.mojopkg` exists after install
- whether `mojo_exact_cpu` works after install

## Validation Results

### `pip install kayak`

Environment:
- fresh Python `3.11` virtual environment

Observed:
- installed package version: `0.1.1`
- `import kayak` worked
- `numpy_reference` scoring worked
- bundled artifact path `kayak_bridge/_artifacts/kayak.mojopkg` was absent
- `mojo_exact_cpu` failed with:
  `RuntimeError: Kayak could not find Mojo sources or a bundled kayak.mojopkg artifact.`

Conclusion:
- the published package works for the documented NumPy reference path
- the published package does not currently ship a working Mojo backend path

### `uv add kayak`

Supported-path environment:
- fresh UV project constrained to Python `>=3.11,<3.12`

Observed:
- installed package version: `0.1.1`
- runtime Python version: `3.11.13`
- `import kayak` worked
- `numpy_reference` scoring worked
- bundled artifact path `kayak_bridge/_artifacts/kayak.mojopkg` was absent
- `mojo_exact_cpu` failed with the same missing-artifact runtime error

Conclusion:
- `uv add kayak` is verified for the reference backend when the consumer project
  is pinned to a supported Python version
- it is not verified for `mojo_exact_cpu`

Additional exploratory note:
- in a separate fresh UV project that only declared `requires-python = ">=3.11"`,
  UV selected Python `3.13.5` on the validation machine
- that install still completed, but it is outside Kayak's declared support range,
  so the supported UV recommendation should remain explicit about Python `3.11`

### `pixi add --pypi kayak`

Environment:
- fresh Pixi project
- `python=3.11`

Observed:
- installed package version: `0.1.1`
- `import kayak` worked
- `numpy_reference` scoring worked
- bundled artifact path `kayak_bridge/_artifacts/kayak.mojopkg` was absent
- `mojo_exact_cpu` failed with the same missing-artifact runtime error

Conclusion:
- the PyPI-backed Pixi install path works for `numpy_reference`
- it does not currently work for `mojo_exact_cpu`

### `pixi add kayak`

Environment:
- fresh Pixi project
- `python=3.11`

Observed:
- dependency resolution failed
- Pixi reported: `No candidates were found for kayak`

Conclusion:
- plain `pixi add kayak` is not currently supported because there is no conda
  package published for `kayak`

### `pixi add python=3.11 mojo` plus `pixi add --pypi kayak`

Environment:
- fresh Pixi project
- `python=3.11`
- `mojo`

Observed:
- installed package version: `0.1.1`
- bundled artifact path `kayak_bridge/_artifacts/kayak.mojopkg` was absent
- `mojo_exact_cpu` still failed with the same missing-artifact runtime error

Conclusion:
- the current failure is not just "consumer does not have Mojo"
- the published package itself is missing the bundled Mojo artifact needed by
  the backend

## Overall Conclusion

Verified today:
- published `kayak 0.1.1` is usable through `pip install kayak`,
  `uv add kayak` with a supported Python `3.11` project, and
  `pixi add --pypi kayak` for `numpy_reference`

Not verified today:
- `mojo_exact_cpu` from the published package
- plain `pixi add kayak`

The immediate packaging gap is concrete:
- the published package does not include `kayak_bridge/_artifacts/kayak.mojopkg`
