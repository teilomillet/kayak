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

- Omar Khattab "Late Interaction in 2030" workshop talk transcript:
  - user-provided transcript in repo discussion on `2026-04-12`
  - argues that late interaction should be treated as a paradigm of local
    interaction plus sublinear search, not as one fixed model family
  - argues that hard-recall retrieval tasks are likely underrepresented when
    the field assumes retrieve-and-rerank up front
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
- the new synthetic faithfulness frontier shows that on the deterministic scale
  slice:
  - `document_proxy` and `centroid_postings` can retain exact-reference recall
    at very small `candidate_k`
  - tighter head-capped variants can cut stage-1 storage much harder, but can
    also collapse faithfulness
- the new BrowseComp-Plus gold frontier shows that on the current `90`-document
  public slice:
  - `document_proxy` is the cheapest full-recall point that was measured
  - the current centroid-family variants do not beat exact latency there once
    they approach full recall
  - judged `nDCG@10` can exceed the exact baseline before exact-reference
    candidate recall reaches `1.0`
- the `binary_f16_le` packed-index payload now has a measured storage result on
  BrowseComp gold:
  - persisted bytes/vector dropped from about `512.08` to `256.08`
  - measured retrieval quality did not change on that slice
- the direct vectors/document pruning benchmark now shows that naive
  prefix-pruning does **not** support a `sqrt(m)`-style budget on BrowseComp
  gold
- the local clause-text ceiling benchmark now shows that a richer text-aware
  path can beat exact MaxSim on BrowseComp gold, but only at much higher
  latency

Evidence:
- [docs/traces/2026-04-12_limit_browsecomp_public_slices.md](docs/traces/2026-04-12_limit_browsecomp_public_slices.md)
- [docs/traces/2026-04-12_single_core_faithfulness_frontier.md](docs/traces/2026-04-12_single_core_faithfulness_frontier.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md](docs/traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_storage_encoding.md](docs/traces/2026-04-12_browsecomp_plus_gold_storage_encoding.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md](docs/traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md](docs/traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md)

### Inferences we are making

- The highest-leverage missing pieces are storage, candidate generation, and serving boundaries, not another benchmark-specific reranker tweak.
- `kayak` should compete as a late-interaction-native engine, not as a generic vector search wrapper.
- Optional text sidecars matter, but they should follow storage and query-contract design rather than precede it.
- the next benchmark step should measure hard stage-1 recall pressure, not only
  "beat dense retrieval on an easy enough first-stage task"
- the next native-candidate iteration should be justified by a win on a harder
  public-slice frontier, not only by synthetic asymptotics
- the next harder hard-recall family should be selected explicitly instead of
  pretending the current small public slices are already sufficient
- heterogeneous encoder late interaction should be treated as unsupported by
  default unless there is explicit joint training or calibration evidence
- the benchmark ladder should stay explicit:
  - trivial sanity tasks
  - harder public text tasks
  - scalable synthetic hard-recall tasks
  - stronger local ceilings
  - later, code- or multimodal-shaped families through the same stage-aware
    reporting surface
- hosted deployment claims should be justified with measured storage, latency,
  and quality numbers rather than with algorithm-level intuition alone
- future paper integrations should be classified before implementation as one
  of:
  - encoder or model substitution
  - document-representation transform
  - search-native sidecar
  - stage-1 engine family
  - stage-2 operator
  - stronger evaluation ceiling
- the document-representation-transform seam is now explicit at the sealed
  segment boundary
- transform execution now exists for prefix pruning and token pooling, and the
  first BrowseComp-Plus gold measurement is recorded
- measured result worth preserving:
  - hierarchical factor-`2` token pooling cut stored vectors and on-disk packed
    storage roughly in half while preserving the current gold-slice `nDCG@10`
  - hierarchical factor-`3` improved the tiny-slice task metric while dropping
    exact-reference top-`10` overlap to `0.9`, which confirms that stage-1
    faithfulness and final task metric are different axes
- the next open step on that seam is stage-aware restoration measurement
  against the exact late-interaction oracle, not more provenance plumbing

Related strategic note:
- [docs/late_interaction_2030.md](docs/late_interaction_2030.md)
- [docs/architecture/extensibility_wall.md](docs/architecture/extensibility_wall.md)
- [docs/traces/2026-04-13_document_representation_transform_contract.md](docs/traces/2026-04-13_document_representation_transform_contract.md)
- [docs/traces/2026-04-13_token_pooling_document_transform.md](docs/traces/2026-04-13_token_pooling_document_transform.md)

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
- [x] Add a first-class document-representation-transform contract so pooling,
  pruning, and future document-side transforms do not remain implicit packed
  index variants

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
- [x] Add a real non-default stage-1 generator with exact stage-2 reranking
- [x] Persist one search-native sidecar per sealed segment
- [x] Make vector budget a first-class benchmark axis
- [x] Measure stage-1 recall against a full exact oracle, not against its own shortlist
- [x] Use that evidence to unblock the next heavier native candidate-generation step

Current interpretation:
- `document_proxy` is now the light proxy-vector baseline.
- `centroid_postings` is now the first search-native centroid/posting baseline.
- `centroid_postings_flat` is now the tighter semantics-preserving native
  follow-on for that centroid/posting baseline.
- `centroid_postings_imputed` is now the first heavier WARP-inspired reduction
  baseline.
- `centroid_postings_imputed_flat` is now the tighter layout follow-on for that
  heavier native path.
- the centroid family now has an explicit primitive layer for:
  - scored centroid selection
  - posting-score accumulation
- the current primitive profile says the imputed selection kernel is the next
  CPU hot path, not accumulation
- approximate plans now carry explicit faithfulness policy rather than silently
  pretending to be exact
- The next heavier engine step should be a tighter WARP/GEM-style native
  engine path built on `centroid_postings_imputed_flat`, not another round of
  heuristic reranking on the older layout.

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
- [x] Keep exact `doc_id` filters on native `document_proxy` and centroid stage-1 paths instead of forcing blanket exact fallback
- [x] Extend native candidate generation from exact `doc_id` pushdown to broader metadata-aware filter selectivity
- [x] Extend native filter-aware candidate generation to real tenant-aware selectivity on shared hosted layouts
- [x] Extend logical-scope pushdown from filtered searches to explicit shared-pool `match_all` serving without relying on tenant-rooted paths alone

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
- [x] Add structured benchmark output JSON for all public benchmark entrypoints
- [x] Add a query explain/profile command for one query against one collection
- [x] Add stage-level counters and histograms
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
- [x] Keep HTTP/JSON simple first
- [x] Expose internal stage/profiling data directly in debug mode

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
- [x] Prototype one compressed token format behind a non-default flag

## Priority 7: GPU And Distributed Execution

Important, but not the first bottleneck today.

Decision:
- CPU remains the correctness and observability anchor
- GPU and distributed work should target the stabilized stage contracts, not bypass them

Immediate TODOs:
- [x] Keep backend boundaries explicit in `runtime/`
- [x] Design stage interfaces so GPU kernels can replace CPU kernels cleanly
- [x] Delay distributed sharding design until segment and tenancy contracts are settled

## Priority 8: Harder Task Ladder And Stronger Ceilings

This is the next strategic evaluation priority after the current stage and
hosted-engine substrate.

Reason:
- the 2030 talk argues that the field undersamples genuinely hard retrieval
  tasks because it assumes retrieve-and-rerank up front
- the Q&A makes the point sharper:
  - text is not the only future modality
  - code and beyond-text retrieval are plausible high-value directions
  - stronger ceilings must be measured and labeled honestly

Decision:
- Kayak should keep one explicit benchmark ladder rather than accumulating
  disconnected evals

Required properties:
- one trivial sanity family
- one harder public text family
- one scalable synthetic hard-recall family
- one stronger local ceiling path with explicit cost accounting
- later, one code- or multimodal-shaped family through the same stage-aware
  reporting surface

Immediate TODOs:
- [ ] Write a benchmark-ladder note with explicit exit criteria per family
- [ ] Add one code- or long-document-shaped hard-recall slice through the
  existing stage-aware JSON pipeline
- [ ] Keep stronger-ceiling artifacts labeled by the actual path used
- [ ] Avoid calling local clause-text ceilings "cross-encoder" or
  "long-context" ceilings

## Priority 9: Encoder Boundary And Interoperability

This is important because the Q&A makes clear that model-agnostic late
interaction is not something we should assume.

Reason:
- arbitrary heterogeneous encoder interactions are likely unsound without
  explicit joint training or calibration
- Kayak is a hosted engine and needs to be honest about what one collection or
  one search space actually means

Decision:
- encoder identity is a first-class engine contract
- heterogeneous late interaction is a research feature, not a default product
  assumption

Required properties:
- explicit encoder identity in collection and segment metadata
- explicit refusal to mix incompatible late-interaction spaces by accident
- room for future calibrated or jointly trained interoperability without
  rewriting the engine

Immediate TODOs:
- [ ] Add one design note for multi-encoder interoperability and non-goals
- [ ] Keep collection and snapshot contracts explicit about one encoder space
- [ ] Add one negative test or validation path that rejects unsound mixed-model
  search assumptions if the current code surface permits them

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

## Phase F: Hosted Collection Loop

Goal:
- make the service and storage contracts executable end to end

Deliverables:
- create collection
- ingest and upsert documents
- delete documents
- snapshot and restore
- exact search and explain against a hosted collection

Exit criteria:
- an external user can run one hosted exact late-interaction collection without
  benchmark-only fixtures

Reason:
- the repo already has the contract surface
- the next step is to make that surface real rather than keep expanding design
  notes

## Phase G: Native Stage-1 Candidate Generation

Goal:
- move from exact-only serving to an explicit stage-1 plus stage-2 engine

Deliverables:
- exact full-scan candidate generator
- one pruning or approximate candidate generator
- candidate-set tracing wired into `SearchPlan`
- recall reporting against exact final results

Exit criteria:
- every result set can report which stage produced which candidates and what
  stage-1 recall it achieved against exact

Reason:
- this is the point where Kayak begins to compete as an engine rather than only
  as an exact scorer
- this is also the point where hard-recall benchmark claims become honest

## Phase H: Hard-Recall Evaluation

Goal:
- test the engine on tasks where weak stage-1 recall is the real bottleneck

Deliverables:
- one or two harder public or synthetic tasks selected for low-recall pressure
- ceiling comparisons against a much more expensive path when justified
- benchmark outputs that separate:
  - stage-1 recall
  - final retrieval quality
  - latency and storage cost

Exit criteria:
- Kayak can explain why its stage design matters on tasks where reranking alone
  cannot rescue poor first-stage recall

Reason:
- the repo should not keep expanding benchmark scope casually
- but once stage contracts exist, harder-recall evaluation becomes the right
  place to test whether the engine is actually ambitious enough

## Next-Cycle TODOs

These are the best next moves right now.

- [x] Wire `kayak/service/` collection and snapshot requests to the serving
  storage layer
- [x] Add one end-to-end hosted collection smoke path:
  create, ingest, snapshot, search, explain
- [x] Execute exact search through an explicit `SearchPlan` runtime path rather
  than benchmark-specific orchestration only
- [x] Implement one tighter WARP/GEM-style native candidate engine behind
  `SearchPlan`
- [x] Extend stage-1 recall reporting to compare the native engine against both
  `document_proxy` and `centroid_postings`
- [x] Write one benchmark-selection note for hard-recall tasks before adding
  more benchmark families

These substrate items are complete.

They are not the same thing as proving the stronger late-interaction
efficiency thesis.

That thesis now moves to a parallel roadmap:
- [docs/late_interaction_efficiency_roadmap.md](docs/late_interaction_efficiency_roadmap.md)

## Parallel Track: Efficiency And Scaling Thesis

Status key:
- `Verified substrate`: the repo already has the machinery needed
- `Instrumented but unproven`: the repo can measure the claim, but has not yet
  established it
- `Unverified claim`: the repo does not yet have sufficient evidence

Current status:
- `Verified substrate`: explicit stage-aware search plans, hosted collection
  loop, stage-aware benchmark output, storage byte accounting
- `Instrumented but unproven`: native stage-1 generators, public hard-recall
  benchmark comparisons, exact-reference candidate recall
- `Unverified claim`: single-core multi-billion-token latency, `6 bytes/vector`
  compression, `sqrt(m)` vector-count laws, stronger ceiling comparisons

Parallel TODOs:
- [x] Add one single-core scaling benchmark over increasing corpus sizes
  - Evidence:
    [docs/traces/2026-04-12_single_core_scale.md](docs/traces/2026-04-12_single_core_scale.md)
- [x] Add one compressed-token benchmark that reports bytes/vector explicitly
- [x] Add one vectors/document sweep that tests aggressive document-vector
  reduction
- [x] Add one asymptotic scaling benchmark for native candidate engines
  - Evidence:
    [docs/traces/2026-04-12_single_core_scale.md](docs/traces/2026-04-12_single_core_scale.md)
- [x] Add one benchmark-selection note for a harder-recall family beyond the
  current default public slices
- [x] Add one stronger-ceiling comparison only after that path exists locally
- [x] Add one scalable synthetic hard-recall family with explicit vector-count
  control and exact-reference candidate recall reporting

## What We Should Not Do Next

- [ ] Do not spend the next cycle tuning the BrowseComp clause-text heuristic
- [ ] Do not add many more benchmark families before the serving storage contract exists
- [ ] Do not hide late-interaction specifics behind generic vector DB abstractions
- [ ] Do not make GPU work the next milestone before storage and stage contracts settle
- [ ] Do not treat "beats dense retrieval" as a sufficient evaluation story by
  itself
- [ ] Do not claim the Omar-style efficiency story before local scaling numbers
  exist

## Success Metrics

We should use these metrics to decide whether the roadmap is working.

Engine metrics:
- exact-search latency by query vector count and document vector count
- candidate-generation latency and recall against exact final results
- storage bytes per document, per token, and per tenant
- ingest throughput and compaction overhead
- snapshot and restore duration
- single-core scaling curves against document count and token count
- bytes per vector for compressed token formats
- retrieval quality versus vectors per document retained

Retrieval metrics:
- `nDCG@10`
- `MRR@10`
- `recall@10`
- candidate recall at stage 1
- candidate recall at fixed budgets such as `100` and `1000`
- final quality versus a more expensive reference path when that comparison is
  justified
- stage-1 recall and final quality on harder-than-current hard-recall tasks

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
- turns a qualitative efficiency claim into a measurable benchmark

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
