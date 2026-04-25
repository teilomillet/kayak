# Data Flow And I/O

Status: current architecture note  
Date: `2026-04-13`

This note defines how data should enter Kayak, what Kayak digests internally,
and what Kayak should return to callers.

It exists to answer a practical question:

- what does a user actually send to Kayak
- what does Kayak store and transform
- what does Kayak return

Related notes:
- [service_api.md](service_api.md)
- [search_plan_semantics.md](search_plan_semantics.md)
- [trust_boundary.md](trust_boundary.md)
- [multi_encoder_interoperability.md](multi_encoder_interoperability.md)

## Core Decision

Kayak's canonical external boundary should be:

- explicit late-interaction representations
- explicit collection metadata about the encoder space
- explicit search plans or planner requests

Not:
- a generic vector-database row schema
- a mandatory raw-document parsing service
- a storage bridge whose semantics are inherited from another database

Reason:
- Kayak's main value is that late interaction stays first-class all the way
  from local code to hosted search
- if the canonical boundary becomes "whatever LanceDB or another vector store
  happens to store", Kayak loses the strongest part of its own architecture

Product-boundary consequence:

- Kayak starts at encoded late-interaction representations, or at plain text
  passed through an explicit caller-selected encoder
- Kayak does not own OCR, PDF parsing, table extraction, handwriting recovery,
  or application answer generation as canonical responsibilities

Reason:

- those document-intelligence steps are real production systems in their own
  right
- the current repo verifies retrieval contracts, storage, planning, scoring,
  and search-service behavior, not raw-document understanding
- keeping this boundary explicit lets performance and correctness work optimize
  the search layer without inheriting unrelated ingestion claims

## Two Interaction Modes

Kayak has two main interaction modes.

### 1. Local SDK

This is the `import kayak` Python path.

The user works with:
- `LateQuery`
- `LateDocuments`
- `LateIndex`
- `SearchPlan`

This mode is for:
- local scoring
- local search
- layout conversion
- candidate-window rescoring
- artifact preparation before upload

### 2. Hosted Engine

This is the collection and snapshot service path.

The user works with:
- `CreateCollectionRequest`
- `UpsertDocumentsRequest`
- `CreateSnapshotRequest`
- `SearchRequest` or `PlannedSearchRequest`

This mode is for:
- hosted multi-vector search
- tenant isolation
- snapshots
- search planning
- explain and debug output

## Canonical Inputs

### Collection creation

The first input is collection metadata:

- `collection_id`
- `tenant_id`
- `namespace_id`
- `model_name`
- `vector_scalar_name`
- `vector_dim`
- layout and sidecar policy

Reason:
- Kayak must know the encoder space up front
- one collection is one late-interaction space, not a bag of unrelated vector
  encoders

### Document ingest

The canonical ingest unit is `UpsertDocument`.

It contains:
- one `EncodedDocument`
- optional `text`
- optional metadata updates

`EncodedDocument` contains:
- `doc_id`
- ragged token vectors
- derived `vector_dim`
- derived `vector_count`

Important current fact:
- the encoder boundary is external
- Kayak currently expects encoded late-interaction vectors, not raw text that
  Kayak itself must encode

### Snapshot publication

Draft data is not queried directly as the canonical search surface.

The user publishes a snapshot explicitly.

That is the boundary where Kayak seals:
- packed exact storage
- optional search-native sidecars
- optional text and metadata sidecars

Reason:
- snapshots are the stable search boundary
- the repo is intentionally explicit about draft versus published state

### Search

The canonical hosted search input is `SearchRequest` or
`PlannedSearchRequest`.

Today that means:
- collection identity
- snapshot identity
- one `EncodedQuery`
- one explicit `query_model_name`
- optional `query_text`
- one `FilterExpression`
- either:
  - one explicit `SearchPlan`
  - or one planner request

Important detail:
- `query_model_name` is explicit because the service encoder boundary is
  external
- that lets the hosted path reject wrong-model queries instead of pretending
  that same-dimension vectors are interchangeable

## What Kayak Digests Internally

Hosted ingest and search currently pass through these stages:

1. Draft mutation log
   - upserts and deletes are appended to draft state

2. Snapshot seal
   - the draft is sealed into one or more segment roots
   - exact packed storage is written
   - configured sidecars are built

3. Snapshot publication
   - one snapshot becomes the explicit visible search boundary

4. Snapshot resolution
   - the runtime loads only the artifacts required by the plan and filter shape

5. Stage-1 candidate generation
   - exact or approximate candidate generation runs

6. Stage-2 reference scoring
   - exact late interaction runs when the plan requires it

7. Stage-3 verification
   - optional text-aware verification can run if the plan requests it

This sequence is already visible in:
- [service_api.md](service_api.md)
- [search_plan_semantics.md](search_plan_semantics.md)

## Canonical Outputs

### Ordinary search

The minimum output should be:
- `doc_id`
- score
- plan metadata

That is `SearchResponse`.

This is the default product boundary for retrieval:
- Kayak returns ids and scores
- the caller can hydrate raw documents on its own side if desired

### Debug search

The richer output is:
- normal search response
- explain payload with stage-by-stage details

That is `DebugSearchResponse`.

Use it when the caller needs:
- candidate provenance
- stage latency and counts
- faithfulness and recall-oriented diagnostics

### Explain-only

The explain path returns:
- a collection-scoped explain object without requiring the ordinary search
  payload as the primary product surface

That is `ExplainResponse`.

## Recommended Integration Shapes

Kayak should support three integration shapes.

### Shape A: Encoded vectors in, ids and scores out

This should be the default hosted boundary.

The caller owns:
- raw documents
- encoding
- document hydration

Kayak owns:
- packed multi-vector storage
- search
- explain
- snapshots and planning

This is the narrowest and safest service boundary.

### Shape B: Encoded vectors plus optional text or metadata

Use this when:
- metadata filtering is needed
- a text-aware verifier is enabled

Important:
- optional text sidecars should stay explicit
- they should not become a silent requirement for ordinary vector-only search

### Shape C: Prebuilt artifact import

For bulk or offline pipelines, Kayak should accept:
- snapshot bundle import
- future upload of packed or sidecar artifacts through explicit adapters

Reason:
- large customers may want to build artifacts outside the live service loop

## ColBERT-Shaped Input Vs LanceDB Bridge

This is the recommended design choice.

### Canonical input should be late-interaction-native

That means:
- ColBERT-shaped ragged token-vector inputs
- or Kayak-native packed artifact inputs

Not:
- LanceDB tables as the core storage contract

Reason:
- ColBERT-shaped encoded documents already match Kayak's exact storage and
  scoring semantics
- LanceDB or another storage system can be useful as an adapter, but it should
  not define Kayak's core meaning of a query, document, or segment

### Where a LanceDB bridge could still make sense

A LanceDB-style bridge can still be useful as:
- a customer-side preprocessing or staging layer
- an import or export adapter
- a metadata or bulk-transfer convenience layer

But it should remain:
- an adapter
- not the canonical data model

Decision:
- Kayak should keep its own late-interaction objects as the source of truth
- external databases may bridge into or out of that boundary

## Multiple Models

Current rule:
- one collection equals one encoder space

Therefore:
- mixing multiple unrelated late-interaction models inside one collection is
  unsupported by default
- separate models should use separate collections or separate snapshots unless
  a future calibrated interoperability story is implemented explicitly

For the detailed rule set, see:
- [multi_encoder_interoperability.md](multi_encoder_interoperability.md)

## End-To-End Shape Today

The current hosted example already follows the intended sequence:

1. create collection
2. upsert encoded documents
3. create snapshot
4. search with an encoded query
5. return search or debug JSON

See:
- [examples/hosted_collection_smoke.mojo](../../examples/hosted_collection_smoke.mojo)

## Final Recommendation

Kayak should standardize on this mental model:

- users bring encoded late-interaction representations
- Kayak digests them into explicit collection and snapshot artifacts
- Kayak returns ids, scores, and optional explain data
- raw-document storage and generic external database schemas remain optional
  integration layers, not the canonical core
