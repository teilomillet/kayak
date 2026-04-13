# Service API Architecture

Status: `Phase A/Phase E bridge draft`  
Date: `2026-04-12`

This document defines the public service contract that `kayak` should expose
before any HTTP transport is implemented.

It is intentionally epistemic:
- verified statements are tied to the current repository
- open choices remain open instead of being hidden in provisional code
- the document distinguishes the canonical typed contract from any future wire format

## Verified Starting Point

These statements are checked against the current repository.

1. Hosted collection identity and storage contracts now exist under
   [`kayak/collections/`](../../kayak/collections).
2. Search planning and explain data now exist under
   [`kayak/planning/`](../../kayak/planning), including:
   - explicit `SearchPlan`
   - explicit `CandidateSet`
   - explicit collection-scoped explain output
3. The repository still does not have an HTTP transport layer.
4. The current explain example,
   [`examples/scifact_collection_explain.mojo`](../../examples/scifact_collection_explain.mojo),
   already demonstrates the core engine shape that a service should wrap:
   - resolve one collection snapshot
   - execute one explicit search plan
   - emit structured explain data

## Decision

Define a canonical service contract in code before choosing a transport.

This is the sound order because:
- the engine already has collection and search-plan primitives
- the public surface should be stable even if HTTP, gRPC, or another transport
  changes later
- transport-specific JSON choices should be a projection of typed engine
  requests and responses, not the source of truth

The canonical typed boundary lives in [`kayak/service/`](../../kayak/service).
The repository now also includes a thin JSON projection layer in
[`kayak/service/json.mojo`](../../kayak/service/json.mojo) so the initial
HTTP/JSON adapter can stay simple and reuse the typed contracts directly.

## Scope

The minimum service surface is:
- create collection
- update collection retention policy
- upsert documents
- delete documents
- create snapshot
- export snapshot
- import snapshot
- collection lifecycle report
- build reclaim plan
- execute reclaim plan
- search
- explain search
- health
- metrics

Out of scope for this step:
- auth
- binary ingest transport
- filter-expression grammar
- streaming result transport
- distributed routing

Those remain separate TODO items on purpose.

## Canonical Typed Contracts

The typed request and response objects are deliberately small.

Collection and mutation requests:
- [`CreateCollectionRequest`](../../kayak/service/collection_requests.mojo)
- [`UpdateCollectionRetentionPolicyRequest`](../../kayak/service/collection_requests.mojo)
- [`UpsertDocument`](../../kayak/service/document_requests.mojo)
- [`UpsertDocumentsRequest`](../../kayak/service/document_requests.mojo)
- [`DeleteDocumentsRequest`](../../kayak/service/document_requests.mojo)

Snapshot requests:
- [`CreateSnapshotRequest`](../../kayak/service/snapshot_requests.mojo)
- [`ExportSnapshotRequest`](../../kayak/service/snapshot_requests.mojo)
- [`ImportSnapshotRequest`](../../kayak/service/snapshot_requests.mojo)

Search and explain:
- [`SearchRequest`](../../kayak/service/search_contracts.mojo)
- [`SearchResponse`](../../kayak/service/search_contracts.mojo)
- [`DebugSearchResponse`](../../kayak/service/search_contracts.mojo)
- [`PlannedSearchRequest`](../../kayak/service/search_contracts.mojo)
- [`PlannedSearchResponse`](../../kayak/service/search_contracts.mojo)
- [`PlannedDebugSearchResponse`](../../kayak/service/search_contracts.mojo)
- [`PlannedExplainResponse`](../../kayak/service/search_contracts.mojo)
- [`ExplainRequest`](../../kayak/service/search_contracts.mojo)
- [`ExplainResponse`](../../kayak/service/search_contracts.mojo)

Filter grammar:
- [`FilterExpression`](../../kayak/filters/expression.mojo)
- [`FilterClause`](../../kayak/filters/clause.mojo)
- [`FilterTerm`](../../kayak/filters/term.mojo)

Health and metrics:
- [`ServiceHealthStatus`](../../kayak/service/service_status.mojo)
- [`ServiceMetricsSnapshot`](../../kayak/service/service_status.mojo)

Lifecycle and reclaim:
- [`CollectionLifecycleRequest`](../../kayak/service/lifecycle_contracts.mojo)
- [`CollectionLifecycleResponse`](../../kayak/service/lifecycle_contracts.mojo)
- [`BuildReclaimPlanRequest`](../../kayak/service/lifecycle_contracts.mojo)
- [`BuildReclaimPlanResponse`](../../kayak/service/lifecycle_contracts.mojo)
- [`ExecuteReclaimRequest`](../../kayak/service/lifecycle_contracts.mojo)
- [`ExecuteReclaimResponse`](../../kayak/service/lifecycle_contracts.mojo)
- [`UpdateCollectionRetentionPolicyResponse`](../../kayak/service/lifecycle_contracts.mojo)

## Why The Search Contract Is Typed First

`SearchRequest` carries a fully materialized `SearchPlan`.

That is deliberate.

The engine already knows how to reason about:
- candidate generation
- explicit candidate budgets
- exact late interaction
- explain data

So the stable internal contract should name those things directly.

Additional guardrail now present in the typed contract:
- `SearchPlan` carries an explicit faithfulness policy
- exact plans default to `exact_stage1_required`
- non-exact plans must declare whether they are `best_effort` or require
  exact-oracle full recall
- the service contract rejects a non-exact plan with
  `oracle_full_recall_required` unless the caller asks for a verifiable
  debug/explain path

Inference:
- a future HTTP layer may accept a simpler body such as `{ "k": 10 }`
  and translate it into `default_exact_search_request(...)`
- but the service implementation itself should operate on the explicit typed
  contract, not on loosely structured JSON dictionaries

## Explicit Planner Layer

The repository now also has an explicit planner-facing search contract:

- [`SearchPlanSelectionRequest`](../../kayak/planning/planner.mojo)
- [`SearchPlanSelection`](../../kayak/planning/planner.mojo)
- [`PlannedSearchRequest`](../../kayak/service/search_contracts.mojo)

This layer is intentionally **above** `SearchPlan`, not a mutation of it.

Reason:
- the repo already treats hidden backend auto-selection as a product risk
- benchmark and debug surfaces need to show which stage-1 path was actually
  chosen
- stage-1 architecture is still evolving, so silent dispatch would freeze
  today's heuristics too early

Current verified behavior:
- the planner inspects a snapshot-scoped artifact inventory derived from sealed
  segment manifests
- the planner chooses a concrete `SearchPlan`
- the chosen plan is returned explicitly in the selection payload and again in
  the executed search/explain response
- `PlannedSearchRequest` can carry:
  - `query_text` for text-family stage-3 verifier overrides
  - `stage2_reference_kind` to override only the stage-2 reference operator
  - `stage3_verifier_kind` to override only the stage-3 verifier
  - `stage2_operator_kind` as a compatibility override when callers still use
    the legacy combined stage-2 name
- when exact fallback is selected, the planner preserves the requested
  `candidate_k` window instead of collapsing it to `final_k`, so a later
  stage-2 override can still rerank the intended exact candidate set

Current guardrails:
- exact `doc_id` filters stay native for `document_proxy` and centroid stage 1
- metadata filters stay native for `document_proxy` and centroid stage 1 when
  every segment has a `document_filter_index` sidecar; older snapshots without
  that sidecar still fall back to exact stage 1
- `gem_graph` currently supports only `match_all` filters
- `exact_stage1_required` falls back to exact stage 1
- `oracle_full_recall_required` without `debug_mode` falls back to exact stage 1

Current default-order claim is intentionally narrow:
- the planner exposes goal-shaped default orders
- callers can override that with explicit preferred generator kinds
- the default order does **not** silently promote `gem_graph`,
  `centroid_postings_head_auto`, or `centroid_postings_blockmax`
  as universal winners, because the current local traces do not justify that

## Proposed HTTP/JSON Projection

The initial transport should stay simple and boring.

Recommended first projection:

```text
POST /v1/collections
PUT  /v1/collections/{tenant}/{namespace}/{collection}/retention
POST /v1/collections/{tenant}/{namespace}/{collection}/documents:upsert
POST /v1/collections/{tenant}/{namespace}/{collection}/documents:delete
POST /v1/collections/{tenant}/{namespace}/{collection}/snapshots
POST /v1/collections/{tenant}/{namespace}/{collection}/snapshots/{snapshot}:export
POST /v1/collections/{tenant}/{namespace}/{collection}/snapshots:import
GET  /v1/collections/{tenant}/{namespace}/{collection}:lifecycle
POST /v1/collections/{tenant}/{namespace}/{collection}/reclaim:plan
POST /v1/collections/{tenant}/{namespace}/{collection}/reclaim:execute
POST /v1/collections/{tenant}/{namespace}/{collection}/search
POST /v1/collections/{tenant}/{namespace}/{collection}/search:explain
GET  /healthz
GET  /metrics
```

Why this is the current recommendation:
- it stays collection-scoped instead of introducing a cross-collection query
  language too early
- it keeps tenant and namespace explicit in the request path
- it lets search operate on an explicit snapshot id instead of a hidden
  mutable "active collection state"

## Search Semantics

The service should search a specific snapshot, not an implicit latest snapshot.

Reason:
- the current collection resolver intentionally requires an explicit
  `SnapshotId`
- the collection manifest does not yet record a canonical active snapshot
- silently switching the visible snapshot under a user would make benchmarking,
  debugging, and reproducibility harder

Therefore:
- `SearchRequest` requires `snapshot_id`
- `SearchRequest` now also carries a typed `FilterExpression`, defaulting to
  `match_all`
- `SearchResponse` echoes that same `snapshot_id`
- `DebugSearchResponse` and `ExplainResponse` attach plan and stage details to
  that exact snapshot

## Debug Mode

The service boundary should expose profiling-oriented data in debug mode, not
hide it behind a separate private implementation surface.

Current design choice:
- ordinary search returns `SearchResponse`
- debug search returns `DebugSearchResponse`
- dedicated explain returns `ExplainResponse`
- the JSON projection now maps `DebugSearchResponse` directly to a payload that
  contains both `"search"` and `"debug"` objects, with the debug object reusing
  the existing `CollectionSearchExplain` JSON projection

Why this is sound:
- it reuses the existing `CollectionSearchExplain` object directly
- it avoids inventing a second explain schema just for the service layer
- it keeps the no-debug search path smaller for ordinary callers

## Ingest Semantics

The service contracts currently assume the encoder boundary is external.

That matches the current repo shape:
- `kayak` already treats encoded queries and encoded documents as stable inputs
- Python/Hugging Face encoding can remain outside the search service
- the hosted engine focuses on storage, snapshots, planning, and search

Inference:
- an HTTP/JSON v0 may temporarily accept encoded vectors directly for
  correctness and portability
- a later ingest transport can add binary payloads or multipart upload without
  changing the canonical typed contracts

## Retention And Lifecycle Semantics

The repository now has an explicit service-level lifecycle control surface.

Verified behavior:
- collection manifests persist `default_keep_latest_inactive_count`
- collection manifests also persist a collection-scoped
  `search_artifact_build_policy` for default stage-1 sidecar construction
- each build spec may now carry a generic per-family `config` payload; the
  service contract persists that payload without promoting any single family's
  schema into the collection core
- the stored default is collection-scoped, not snapshot-scoped
- lifecycle-report and reclaim-plan requests can optionally carry an ephemeral
  `SnapshotRetentionPolicy` override
- when no override is present, lifecycle and reclaim operations derive their
  effective policy from the collection manifest
- stage-1 snapshot loading and filter support are derived from explicit
  stage-1 capability contracts rather than from ad hoc generator conditionals
- reclaim execution still requires an explicit plan, which preserves the
  previous stale-plan guardrail instead of introducing implicit background
  deletion

This choice is intentional:
- a collection default belongs with collection continuity and hosted-engine
  operations
- default stage-1 sidecar selection belongs there too, because sealed segments
  should inherit stable build intent from the collection rather than from
  whatever the current seal helper happens to hard-code
- richer family-specific knobs should remain payloads behind that registry
  boundary, not become new top-level collection fields
- pinned-snapshot overrides remain request-scoped until there is a stronger
  reason to persist them
- plan-then-execute remains the sound default because it keeps deletion
  behavior auditable and testable

## Open Choices

These remain intentionally undecided:
- auth and per-tenant credentials
- metadata encoding
- exact filter-evaluation storage layout
- write-ahead log or ingest-buffer protocol
- snapshot archive format
- wire-level representation for vectors
- metrics exposition format beyond the typed `ServiceMetricsSnapshot`

## Immediate Follow-On Work

The next service-adjacent work should be:

1. add a minimal HTTP adapter that translates JSON bodies into
   `kayak/service/` typed contracts
2. route debug mode directly to `CollectionSearchExplain`
3. reuse collection storage reports and snapshot-bundle export/import in the
   service layer
4. add an auth and tenant-isolation story once the core request grammar settles
5. wire filter execution into candidate generation and exact-stage pruning

That sequence preserves the current engine contracts and keeps the transport
thin.
