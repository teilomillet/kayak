# 2026-04-13: internal logical-scope filter pushdown

## Claim

Kayak now makes filtered tenant and namespace isolation harder to bypass by
encoding collection, tenant, and namespace scope as reserved internal filter
postings inside new `document_filter_index` sidecars.

## Why this change is justified

- The repository already had native metadata-aware candidate generation, but it
  still treated tenant-rooted paths as the only isolation mechanism.
- That was sufficient for the current tenant-isolated layout, but it was not a
  durable substrate for future shared hosted layouts.
- The sound next step was not a fake shared-pool runtime. It was to make the
  filter substrate itself capable of carrying logical scope.

## Design

- Added reserved internal filter fields for:
  - collection identity
  - tenant identity
  - namespace identity
- New sealed segments now persist those scope postings alongside user metadata
  postings in `document_filter_index`.
- Public request filters and user document metadata are blocked from using the
  reserved scope fields.
- Filtered execution now composes the public filter with internal logical scope
  when the segment sidecar advertises scope postings.
- Exact fallback reuses the same allowlist path when possible, and keeps the
  old metadata runtime behavior for older scope-unaware segments.

## Evidence

Verified locally with:

- `pixi run mojo -I . tests/test_filters.mojo`
  - 6/6 passed
- `pixi run mojo -I . tests/test_document_filter_index.mojo`
  - 4/4 passed
- `pixi run mojo -I . tests/test_search_planner.mojo`
  - 8/8 passed
- `pixi run mojo -I . tests/test_service_contracts.mojo`
  - 22/22 passed
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
  - 31/31 passed
- `pixi run mojo -I . tests/test_service_runtime.mojo`
  - 19/19 passed

Key checks covered:

- conjoined logical-scope filters evaluate correctly against the exact runtime
- new `document_filter_index` artifacts carry internal collection, tenant, and
  namespace postings
- allowlists respect mixed-scope shared-pool-style postings
- public search/planner requests reject reserved internal scope fields
- user metadata updates cannot spoof reserved internal scope keys
- exact, proxy, and centroid execution still pass the existing hosted search and
  planner suites

## Limits

- This change hardens filtered search paths only.
- Shared-pool `match_all` serving still needs an explicit scope path; it is not
  solved by this tranche.
