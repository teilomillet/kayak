# TODO

`kayak` roadmap for a late-interaction-native search engine.

Date written: April 12, 2026

This document is intentionally epistemic:
- claims are tied to sources or to measurements already captured in this repo
- inferences are labeled as such
- roadmap items are prioritized by infrastructure leverage, not by novelty

## Product Scope

`kayak` is not primarily a model training library, generic vector DB, or end-user RAG app.

The product scope is:
- a storage engine for multi-vector document representations
- a query execution engine for late interaction
- a serving system for hosted tenant data
- a narrow client and SDK surface on top of that engine

In plain terms:
- users put their data into `kayak`
- `kayak` stores, indexes, filters, searches, reranks, profiles, and serves it
- the core value is late-interaction-native execution and storage, not generic embeddings infrastructure

## Epistemic Baseline

### Sources checked on April 12, 2026

- ColBERT official repository:
  - https://github.com/stanford-futuredata/ColBERT
  - stable branch explicitly describes `main` as `ColBERTv2 + PLAID`
- ColBERTv2 paper:
  - https://arxiv.org/abs/2112.01488
  - states late interaction is effective but storage-heavy, and reports `6-10x` footprint reduction via residual compression
- PLAID paper:
  - https://arxiv.org/abs/2205.09707
  - reports a two-stage centroid-based engine for late interaction with large CPU and GPU speedups
- WARP paper:
  - https://arxiv.org/abs/2501.17788
  - reports engine-level improvements over XTR and a `3x` speedup over ColBERTv2/PLAID
- GEM paper:
  - https://arxiv.org/abs/2603.20336
  - argues native multi-vector indexing matters and reports up to `16x` speedups over prior methods
- Curator paper:
  - https://arxiv.org/abs/2401.07119
  - argues multi-tenant vector search needs index designs beyond shared-index filtering vs per-tenant isolation extremes
- Qdrant multivector docs:
  - https://qdrant.tech/documentation/manage-data/vectors/
  - official `max_sim` multivector comparator
- Qdrant multivector tutorial:
  - https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/
  - explicitly recommends dense first-stage retrieval plus multivector reranking, and explicitly warns against indexing every token vector with HNSW
- Qdrant MUVERA postprocessing docs:
  - https://qdrant.tech/documentation/fastembed/fastembed-postprocessing/
  - official example of fast first-stage retrieval plus multivector reranking in one query path
- Vespa tensor guide:
  - https://docs.vespa.ai/en/ranking/tensor-user-guide.html
  - demonstrates generic tensor and multi-vector serving infrastructure
- Mixedbread public materials:
  - https://www.mixedbread.com/blog
  - https://www.mixedbread.com/pricing
  - https://www.mixedbread.com/docs/stores/search/rerank
  - official product/docs show hosted ingestion, storage, search, reranking, metadata, and usage-based billing

### Repo measurements already checked

- `LIMIT-small` is nearly solved on the current exact ColBERTv2 slice.
- `BrowseComp-Plus` is materially harder and exposes real ranking pressure.
- For the current gold-hard query `772`, the answer-bearing document is already in the top-`20` candidate window and sits at rank `18`, not deep in the corpus.
- A BrowseComp-only text-sidecar prototype can move that document from rank `18` to rank `4`, improving the gold slice, but it is not yet a globally good default.

Evidence:
- [docs/traces/2026-04-12_limit_browsecomp_public_slices.md](docs/traces/2026-04-12_limit_browsecomp_public_slices.md)

### Inferences we are making

- The highest-leverage missing pieces are storage, candidate generation, and serving boundaries, not another benchmark-specific reranker tweak.
- `kayak` should compete as a late-interaction-native engine, not as a generic vector search wrapper.
- Optional text sidecars matter, but they should follow storage and query-contract design rather than precede it.

## North Star

Build the best infrastructure substrate for late interaction.

Concretely, that means:
- best-in-class exact MaxSim execution on CPU first, GPU next
- storage formats designed around multi-vector retrieval, not bolted onto single-vector assumptions
- candidate generation that is a first-class part of the engine
- multi-tenancy, filters, updates, snapshots, and compaction as engine primitives
- clear explainability and profiling surfaces for search behavior

## What "SOTA" Means Here

For this project, "SOTA" has three layers:

1. Retrieval quality:
   - match or improve strong late-interaction baselines on public retrieval slices
   - do not regress multi-vector semantics into single-vector shortcuts

2. Engine quality:
   - lower latency and storage cost for equivalent retrieval quality
   - predictable ingest, update, and query behavior
   - credible observability and reproducibility

3. Product quality:
   - usable by others without reading the whole codebase
   - stable service and client boundaries
   - operationally sane multi-tenant hosting model

We should not call the system "best" if only layer 1 is strong.

## Non-Goals For Now

- training a new flagship retrieval model inside `kayak`
- trying to be a general-purpose vector DB first
- shipping a generic text reranker before text storage is first-class
- adding many more benchmark slices before storage and query contracts stabilize
- switching the default engine to approximate search before the exact path is deeply profiled and well-explained

## Strategic Priorities

## Priority 1: Retrieval-Native Storage

This is the most important foundation item.

Reason:
- ColBERTv2 explicitly identifies storage footprint as a central obstacle.
- PLAID, WARP, and GEM are all engine papers, not just model papers.
- our own repo already shows that text sidecars, packed indexes, and derived layouts need clearer contracts.

Decision:
- storage is a first-class subsystem, not an implementation detail

Target primitives:
- `Collection`
- `Segment`
- `Posting` or `DocumentEntry`
- `StoredPackedIndex`
- `StoredDocumentTextCorpus` as an optional sidecar
- `SegmentStats`
- `Snapshot`
- `CompactionPlan`

Required properties:
- explicit vector counts
- explicit vector dimensionality and scalar type
- append-friendly ingest path
- immutable search segments after seal
- background compaction
- optional text and metadata sidecars
- format versioning

Immediate TODOs:
- [x] Define a retrieval segment manifest format
- [x] Split "judged task storage" from "serving collection storage"
- [x] Add an explicit optional text-sidecar artifact keyed by `doc_id`
- [x] Add segment-level stats: doc count, token count, vector count, average vectors/doc, byte size
- [x] Add snapshot/export/import boundaries

## Priority 2: Candidate Generation As A First-Class Engine Stage

This is the second most important item.

Reason:
- PLAID, Qdrant MUVERA, and our own BrowseComp trace all support the same conclusion:
  fast candidate generation plus exact late interaction is the dominant serving pattern.
- exact full-corpus MaxSim should stay the correctness anchor, but it should not be the only engine shape.

Decision:
- `kayak` should explicitly model:
  - stage 1 candidate generation
  - stage 2 exact late interaction
  - optional stage 3 reranking

Target primitives:
- `CandidateGenerator`
- `CandidateSet`
- `CandidateBudget`
- `ExactLateInteractionStage`
- `RerankerStage`
- `SearchPlan`

Candidate-generation families to support:
- exact full scan
- centroid/pruning family inspired by PLAID
- approximate proxy vector family such as MUVERA-style first stage
- future native multi-vector index path if we implement something closer to GEM/WARP ideas

Immediate TODOs:
- [x] Define a `SearchPlan` contract that names each stage explicitly
- [x] Add a candidate-set artifact and profiling output
- [x] Add benchmark output for recall of stage 1 against exact final results
- [x] Add candidate-window sweeps as a standard benchmark

## Priority 3: Multi-Tenant Serving And Filter-Aware Retrieval

This is the third most important item.

Reason:
- if users host their data in `kayak`, tenancy is not optional
- Curator is strong evidence that multi-tenant search is an index-design problem, not just a payload filter problem
- the engine should not force users into either "one giant shared index" or "one tiny index per tenant" as its only modes

Decision:
- tenancy and filtering belong in the engine design from the start

Required properties:
- tenant isolation
- filter-aware candidate generation
- efficient small-tenant and large-tenant behavior
- predictable update/delete behavior

Immediate TODOs:
- [x] Define tenant and namespace boundaries in storage manifests
- [x] Define a filter expression model for search requests
- [x] Add per-tenant and cross-tenant segment layout design note
- [x] Add benchmark fixtures for low-selectivity and high-selectivity filters

## Priority 4: Observability, Profiling, And Explainability

This is a low-hanging fruit with large product payoff.

Reason:
- infrastructure users need to understand why a result was returned, where time went, and how much each stage cost
- current traces are useful, but they are mostly human-readable and benchmark-specific

Decision:
- every major engine path should have a machine-readable profile surface

Required outputs:
- search-plan summary
- stage timings
- candidate counts
- vector counts
- byte counts
- reranker usage
- filter selectivity
- per-query trace exports

Immediate TODOs:
- [ ] Add structured benchmark output JSON for all public benchmark entrypoints
- [x] Add a query explain/profile command for one query against one collection
- [ ] Add stage-level counters and histograms
- [x] Add segment-level storage reports

## Priority 5: Service Boundary

This is where the engine becomes usable by others.

Decision:
- the public product boundary should be a search service, not just local library calls

Minimum service API:
- create collection
- ingest documents
- upsert/update documents
- delete documents
- snapshot/export/import
- search
- search with explicit `SearchPlan`
- explain one result set
- health and metrics

Immediate TODOs:
- [x] Write the public API contract before implementing transport details
- [ ] Keep HTTP/JSON simple first
- [ ] Expose internal stage/profiling data directly in debug mode

## Priority 6: Compression And Layout Optimization

This is important, but should come after the main storage/query contracts are clean.

Reason:
- ColBERTv2, PLAID, WARP, and GEM all indicate that storage layout and compression are central to scale
- but compression choices are easier to make once the storage artifacts are explicit

Possible directions:
- residual compression
- PQ or bit-level compression for token vectors
- mixed precision or lower-bit storage
- segment-local centroid summaries
- cold/hot tiering

Immediate TODOs:
- [x] Benchmark byte cost per document and per token as a first-class metric
- [x] Add storage/report tooling before picking a default compression path
- [ ] Prototype one compressed token format behind a non-default flag

## Priority 7: GPU And Distributed Execution

Important, but not the first bottleneck today.

Decision:
- CPU remains the correctness and observability anchor
- GPU and distributed work should target the stabilized stage contracts, not bypass them

Immediate TODOs:
- [ ] Keep backend boundaries explicit in `runtime/`
- [ ] Design stage interfaces so GPU kernels can replace CPU kernels cleanly
- [ ] Delay distributed sharding design until segment and tenancy contracts are settled

## Execution Plan

## Phase A: Engine Contracts

Goal:
- move from "good exact-search codebase" to "clear engine primitives"

Deliverables:
- `TODO`-driven design doc for collection/segment/search-plan objects
- retrieval segment manifest
- optional text-sidecar manifest
- machine-readable benchmark result format

Exit criteria:
- storage and search APIs can be explained without reference to benchmark-specific task types

## Phase B: Serving Storage

Goal:
- make the engine capable of hosting real tenant data

Deliverables:
- collection storage layout
- ingest path
- sealed segments
- compaction plan
- snapshots
- metadata/filter sidecars

Exit criteria:
- a user can ingest, update, delete, snapshot, and search a hosted collection

## Phase C: Candidate Generation

Goal:
- make stage 1 explicit and measurable

Deliverables:
- exact baseline candidate generator
- one approximate or pruning-based generator
- candidate recall benchmarking versus exact full scan

Exit criteria:
- every search result can report which stage produced which candidate set

## Phase D: Text-Aware And Metadata-Aware Reranking

Goal:
- make stage 3 honest and portable

Deliverables:
- first-class text sidecar
- generic reranker interface
- evidence that reranking helps without hidden benchmark leakage

Exit criteria:
- rerankers no longer depend on ad hoc JSON task caches

## Phase E: Productionization

Goal:
- make the engine operable

Deliverables:
- HTTP service
- auth and tenant isolation model
- metrics and tracing
- packaging and deployment story

Exit criteria:
- an external user can run `kayak` as a service and reason about its performance

## Near-Term TODOs

These are the best next moves right now.

- [ ] Write `docs/architecture/segment_storage.md` describing collection, segment, manifest, snapshot, and compaction primitives
- [x] Write `docs/architecture/segment_storage.md` describing collection, segment, manifest, snapshot, and compaction primitives
- [x] Introduce a serving-oriented storage package parallel to benchmark task storage
- [x] Add `StoredDocumentTextCorpus` as an optional artifact
- [x] Define `SearchPlan`, `CandidateGenerator`, and `CandidateSet`
- [x] Emit structured benchmark JSON for the real-slice benchmarks
- [x] Add one profile/explain command for a single query
- [x] Add a minimal service API design doc

## What We Should Not Do Next

- [ ] Do not spend the next cycle tuning the BrowseComp clause-text heuristic
- [ ] Do not add many more benchmark families before the serving storage contract exists
- [ ] Do not hide late-interaction specifics behind generic vector DB abstractions
- [ ] Do not make GPU work the next milestone before storage and stage contracts settle

## Success Metrics

We should use these metrics to decide whether the roadmap is working.

Engine metrics:
- exact-search latency by query vector count and document vector count
- candidate-generation latency and recall against exact final results
- storage bytes per document, per token, and per tenant
- ingest throughput and compaction overhead
- snapshot and restore duration

Retrieval metrics:
- `nDCG@10`
- `MRR@10`
- `recall@10`
- candidate recall at stage 1

Product metrics:
- one-command local service startup
- documented collection ingest and search flow
- reproducible benchmark outputs
- clear tenant isolation semantics

## Decision Rule

When choosing between two roadmap items, prefer the one that:
- strengthens storage or query contracts
- improves stage explainability
- generalizes across datasets and models
- keeps vector counts and late-interaction semantics explicit

Avoid work that:
- only improves one benchmark slice
- hides semantics in opaque heuristics
- makes the system harder to profile or explain

## Current Working Thesis

The best infra for late interaction will not look like a generic vector store with multi-vectors bolted on.

It will look like:
- retrieval-native storage
- explicit multi-stage search planning
- exact MaxSim as a first-class primitive
- optional but honest text and metadata sidecars
- tenant-aware serving and operations

That is the plan this repo should follow.
