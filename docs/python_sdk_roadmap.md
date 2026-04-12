# Kayak Python Roadmap

This roadmap turns the Python SDK charter into a sequence of concrete,
testable steps.

It is intentionally product-first:
- it describes what the Python SDK should become
- it distinguishes SDK work from hosted-engine work
- it uses exit criteria so progress can be validated rather than inferred

For the mission and philosophy behind this plan, see
[docs/python_sdk_charter.md](python_sdk_charter.md).

## Guiding Decision

Build `kayak` as the local Python late-interaction SDK first.

Add any remote or hosted-engine client only as an explicit second layer.

Reason:
- the verified public API already models local late-interaction objects and
  exact operations
- the hosted service boundary is a separate concern with different stability and
  release constraints

## Phase 0: Current Verified Baseline

Current verified state:
- `import kayak` is the supported Python entrypoint
- the public API exposes explicit late-interaction objects and exact operations
- `numpy_reference` works after published installation paths
- `mojo_exact_cpu` is an explicit optional backend
- the package README is package-local rather than inherited from the monorepo
- the SDK now has a product charter and a bounded Ordeal smoke path

Why this matters:
- it means the project already has the substrate for a real SDK
- the next work should stabilize and sharpen that SDK instead of redefining it

## Phase 1: Stabilize The Core SDK

Goal:
- make `kayak` a small, coherent, trustworthy Python programming surface

Scope:
- keep the public API centered on:
  - `LateQuery`
  - `LateDocuments`
  - `LateIndex`
  - `LateScores`
  - `query`
  - `documents`
  - `pack`
  - `to_layout`
  - `maxsim`
  - `search`
- keep `numpy_reference` as the always-available reference backend
- keep `mojo_exact_cpu` explicit and optional
- keep public imports rooted at `import kayak`

Exit criteria:
- the public API contract is covered by tests
- the package README is tailored to Python users
- `pip install kayak` supports the reference path in a fresh Python 3.11
  environment
- `uv add kayak` supports the reference path in a fresh project pinned to the
  supported Python version
- `pixi add --pypi kayak` supports the reference path in a fresh Pixi project
- the docs clearly distinguish the public SDK from internal engine modules

## Phase 2: Improve Ergonomics Without Hiding Structure

Goal:
- make the SDK feel better to use without turning it into implicit magic

Likely additions:
- first-class reranking helpers
- explicit batch helpers where shapes remain readable
- backend capability introspection
- clearer conversion helpers for NumPy and PyTorch inputs
- more runnable examples built around real late-interaction workflows

Constraints:
- no hidden backend switching
- no implicit exact vs approximate dispatch
- no fake broadcasting model that erases ragged late-interaction structure

Exit criteria:
- each new convenience entrypoint still preserves explicit layout and backend
  choices
- examples and tests cover the new ergonomics directly

## Phase 3: Separate SDK And Engine Concerns Cleanly

Goal:
- keep the public Python SDK usable on its own while allowing a richer engine
  product to exist behind it or alongside it

Recommended boundary:
- Kayak Python owns local late-interaction programming
- Kayak Engine owns collections, snapshots, serving, and operational concerns

Potential packaging shapes:
- keep one monorepo, publish only `kayak` publicly
- keep engine internals private or slower-moving
- if a remote client is needed later, add it explicitly under a separate
  namespace or package instead of making it the identity of `kayak`

Reason:
- this preserves the strongest public story: programmable late interaction in
  Python
- it avoids forcing service-level coupling into the core SDK too early

Exit criteria:
- public docs can explain the difference between SDK and engine in one short
  paragraph
- internal engine modules can evolve without changing the `kayak` import
  contract

## Phase 4: Deepen The Fast Path

Goal:
- improve the Mojo-backed path without making it a hard prerequisite for SDK
  adoption

Likely work:
- reduce Python-to-Mojo conversion overhead where measurement justifies it
- expand tested layout-specialized kernels
- keep NumPy as the correctness oracle for optimized paths
- add more exact differential tests between NumPy and Mojo

Constraints:
- no performance claim without measurement
- no optimization that weakens the explicit late-interaction model

Exit criteria:
- optimized paths have matching-reference tests
- performance comparisons are backed by reproducible benchmark commands

## Deferred Until The Service Boundary Is Ready

These are real possibilities, but they should stay out of the core SDK story
until the engine contract stabilizes:
- hosted collection administration
- remote search clients as the main entrypoint
- deployment and tenancy configuration
- service lifecycle and health operations

Reason:
- these concerns belong to the engine product, not the narrow Python SDK

## Acceptance Checklist

The roadmap is on track when all of the following stay true:
- someone can install `kayak` and use it productively in normal Python code
- the reference backend works without Mojo
- Mojo acceleration is explicit rather than magical
- the public docs explain what belongs to the SDK and what belongs to the
  engine
- new API additions make late interaction easier to program against without
  hiding vector counts, layouts, or MaxSim semantics
