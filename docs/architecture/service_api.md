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

## Scope

The minimum service surface is:
- create collection
- upsert documents
- delete documents
- create snapshot
- export snapshot
- import snapshot
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
- [`ExplainRequest`](../../kayak/service/search_contracts.mojo)
- [`ExplainResponse`](../../kayak/service/search_contracts.mojo)

Health and metrics:
- [`ServiceHealthStatus`](../../kayak/service/service_status.mojo)
- [`ServiceMetricsSnapshot`](../../kayak/service/service_status.mojo)

## Why The Search Contract Is Typed First

`SearchRequest` carries a fully materialized `SearchPlan`.

That is deliberate.

The engine already knows how to reason about:
- candidate generation
- explicit candidate budgets
- exact late interaction
- explain data

So the stable internal contract should name those things directly.

Inference:
- a future HTTP layer may accept a simpler body such as `{ "k": 10 }`
  and translate it into `default_exact_search_request(...)`
- but the service implementation itself should operate on the explicit typed
  contract, not on loosely structured JSON dictionaries

## Proposed HTTP/JSON Projection

The initial transport should stay simple and boring.

Recommended first projection:

```text
POST /v1/collections
POST /v1/collections/{tenant}/{namespace}/{collection}/documents:upsert
POST /v1/collections/{tenant}/{namespace}/{collection}/documents:delete
POST /v1/collections/{tenant}/{namespace}/{collection}/snapshots
POST /v1/collections/{tenant}/{namespace}/{collection}/snapshots/{snapshot}:export
POST /v1/collections/{tenant}/{namespace}/{collection}/snapshots:import
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

## Open Choices

These remain intentionally undecided:
- auth and per-tenant credentials
- filter-expression request grammar
- metadata encoding
- write-ahead log or ingest-buffer protocol
- snapshot archive format
- wire-level representation for vectors
- metrics exposition format beyond the typed `ServiceMetricsSnapshot`

## Immediate Follow-On Work

The next service-adjacent work should be:

1. define the filter expression model
2. add a minimal HTTP adapter that translates JSON bodies into
   `kayak/service/` typed contracts
3. route debug mode directly to `CollectionSearchExplain`
4. reuse collection storage reports and snapshot-bundle export/import in the
   service layer
5. add an auth and tenant-isolation story once the core request grammar settles

That sequence preserves the current engine contracts and keeps the transport
thin.
