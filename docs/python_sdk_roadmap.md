# Kayak Python Roadmap

This roadmap turns the Python SDK charter into a sequence of concrete,
testable steps.

It is intentionally product-first:
- it describes what the Python SDK should become
- it distinguishes SDK work from hosted-engine work
- it uses exit criteria so progress can be validated rather than inferred

For the mission and philosophy behind this plan, see
[docs/python_sdk_charter.md](python_sdk_charter.md).

## Current Status

As of `2026-04-12`, the first SDK pass of this roadmap is complete:
- Phase 1 exit criteria are satisfied
- Phase 2 ergonomics additions are implemented and tested
- Phase 3 SDK-versus-engine boundary docs are in place
- Phase 4 has an initial measured fast path for batched Mojo scoring
- Phase 5 exposes a first public stage-aware primitive layer for local search

Future work can still deepen the fast path, but the current roadmap items now
have verified evidence instead of only intent.

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

Status:
- completed on `2026-04-12`

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

Verified evidence:
- [python/tests/test_public_api_contract.py](../python/tests/test_public_api_contract.py)
- [python/tests/test_python_sdk_docs.py](../python/tests/test_python_sdk_docs.py)
- install-path validation recorded in [docs/python_sdk.md](python_sdk.md)

## Phase 2: Improve Ergonomics Without Hiding Structure

Status:
- completed on `2026-04-12`

Goal:
- make the SDK feel better to use without turning it into implicit magic

Likely additions:
- explicit batch helpers where shapes remain readable
- backend capability introspection
- explicit candidate-window subsetting instead of a hidden rerank primitive
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

Verified evidence:
- public exports now include `LateQueryBatch`, `query_batch`, `maxsim_batch`,
  `search_batch`, `available_backends`, and `backend_info`
- [python/tests/test_batch_api.py](../python/tests/test_batch_api.py)
- [python/examples/query_batch.py](../python/examples/query_batch.py)
- [python/examples/backend_info.py](../python/examples/backend_info.py)
- explicit candidate-window subsetting is available through
  `LateIndex.select(...)`

## Phase 3: Separate SDK And Engine Concerns Cleanly

Status:
- completed on `2026-04-12` for the current public SDK boundary

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

Verified evidence:
- [docs/python_sdk_charter.md](python_sdk_charter.md)
- [docs/python_sdk.md](python_sdk.md)
- [python/kayak/__init__.py](../python/kayak/__init__.py)
- [python/kayak_bridge/__init__.py](../python/kayak_bridge/__init__.py)

## Phase 4: Deepen The Fast Path

Status:
- initial fast-path step completed on `2026-04-12`

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

Verified evidence:
- shared-index batch dispatch for `mojo_exact_cpu` reuses the loaded Mojo
  module and precomputed index payload across all queries in a batch
- [python/tests/test_batch_api.py](../python/tests/test_batch_api.py) now
  differentially covers both packed and `hybrid_flat_dim128` Mojo paths
- [python/ordeal_tests/test_python_sdk_chaos.py](../python/ordeal_tests/test_python_sdk_chaos.py)
  exercises batch scoring in both NumPy-only and Mojo-enabled runs
- reproducible benchmark commands now exist:
  - `pixi run bench_python_batch_maxsim_naive_raw`
  - `pixi run bench_python_batch_maxsim_shared_raw`
  - `pixi run bench_python_batch_maxsim_naive`
  - `pixi run bench_python_batch_maxsim_shared`
- recorded trace:
  [docs/traces/2026-04-12_python_sdk_batch_fast_path.md](traces/2026-04-12_python_sdk_batch_fast_path.md)

## Phase 5: Public Stage-Aware Primitives

Status:
- completed on `2026-04-13` for the first local Python pass

Goal:
- expose explicit candidate generation and search plans publicly in Python
  without collapsing them into hidden rerank behavior

Scope:
- `CandidateGenerator`
- `SearchPlan`
- `SearchStageProfile`
- `generate_candidates(...)`
- `search_with_plan(...)`
- first public stage-1 generators:
  - `exact_full_scan`
  - `document_proxy`

Reason:
- the engine already had explicit stage-aware contracts
- the Python SDK was still exact-only at the public surface
- `document_proxy` is the lightest real non-exact stage-1 primitive already
  present in the repo, so it is the least speculative first public lift

Exit criteria:
- the public API exposes an explicit search-plan layer
- tests verify that `document_proxy` candidate generation is distinct from the
  exact stage
- tests verify that exact reranking over the candidate window is explicit and
  stable
- docs explain that richer native generators remain engine-side for now

Verified evidence:
- [python/tests/test_search_plan_api.py](../python/tests/test_search_plan_api.py)
- [python/examples/search_plan.py](../python/examples/search_plan.py)
- [docs/python_sdk.md](python_sdk.md)

## Phase 6: Public Stage-2 Operators

Status:
- proposed next SDK-architecture step

Goal:
- align the public Python SDK with the engine's future stage-2 primitive
  boundary instead of freezing around today's exact-only shortlist rerank path

Reason:
- the current repo now has explicit stage-1 families, but stage 2 is still
  effectively hardcoded as exact late interaction in the Python plan path
- the stronger ceiling and text-aware rerank work already exists in the engine
  tree, but not behind one stable public operator model
- if `kayak` is the canonical Python late-interaction SDK, users should
  program against a stable refinement primitive rather than one specific
  implementation detail

Scope:
- add an explicit Python `Stage2Operator`
- make `search_with_plan(...)` carry that operator
- keep exact late interaction as the default stage-2 operator
- allow richer operators only when their artifact requirements are explicit
- preserve the rule that backend choice stays explicit and non-magical

Constraints:
- no hidden promotion from exact stage 2 to text reranking
- no generic callback API that hides required artifacts
- no public API that forces the SDK to expose hosted-engine lifecycle concerns

Exit criteria:
- the Python search-plan API can express stage 2 without hardcoding
  `"exact_late_interaction"` as the only public refinement mode
- exact late interaction and at least one additional stage-2 operator share the
  same public plan shape
- tests verify that artifact requirements stay explicit in the public API

Architecture note:
- [docs/architecture/stage2_primitives.md](architecture/stage2_primitives.md)

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
