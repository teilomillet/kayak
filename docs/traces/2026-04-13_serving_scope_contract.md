# 2026-04-13: explicit serving-scope contract

## Claim

Kayak now treats serving scope as an explicit search contract instead of an
implicit consequence of tenant-rooted layout.

## Why this change is justified

- Query execution already knew how to conjoin internal logical scope with user
  filters.
- That was not enough because planner selection and snapshot loading were still
  reasoning from the user-visible filter alone.
- In `shared_pool` mode this created a semantic gap:
  - public `match_all` queries still require scoped candidate generation
  - exact `doc_id` queries still require scoped candidate generation
  - a missing scope-aware filter sidecar means there is no safe fallback, not
    “just use exact”

## Design

- Added a typed `SearchServingScope` contract with two current modes:
  - `layout_rooted`
  - `logical_filter_pushdown`
- Planner selection now carries that serving-scope contract explicitly.
- Shared-pool planning now requires `document_filter_index` sidecars on every
  segment even for public `match_all`.
- Shared-pool planning now rejects unsafe exact fallback when no safe generator
  is available.
- Explain and planned-search JSON now surface the serving-scope contract.

## Evidence

Verified locally with:

- `pixi run mojo -I . tests/test_search_planner.mojo`
  - 10/10 passed
- `pixi run test_service_contracts`
  - 22/22 passed
- `pixi run test_service_json`
  - 8/8 passed
- `pixi run test_service_runtime`
  - 22/22 passed
- `pixi run test_collection_search_plan`
  - 31/31 passed
- `pixi run mojo -I . tests/test_planner_benchmark_json.mojo`
  - 3/3 passed
- `pixi run mojo -I . tests/test_planner_evidence_json.mojo`
  - 4/4 passed
- `pixi run test_semantic_guardrails`
  - Python guardrails passed
  - Mojo guardrails passed

Key checks covered:

- planner selection keeps native stage 1 when logical pushdown is available
- planner rejects scope-unaware availability for shared-pool serving
- shared-pool exact `match_all` search now loads the scope filter sidecar
- shared-pool GEM planning falls back to a safe exact path instead of pretending
  GEM is valid under scoped serving
- scope-unaware legacy filter indices are rejected under shared-pool serving

## Limits

- The current serving-scope contract is still collection-scoped.
- It does not yet model richer future scope families beyond the current
  `layout_rooted` vs `logical_filter_pushdown` distinction.
