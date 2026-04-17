# Recent Paper Targets

Status: working paper shortlist  
Date: `2026-04-17`

This note records the recent late-interaction papers that are the best current
fit for `kayak`, plus the first implementation decision.

It is intentionally epistemic:
- paper choices are tied to primary sources checked on `2026-04-17`
- fit is judged against the current repo seams, not only against paper novelty
- the first implementation target is chosen for lowest coupling risk and
  highest benchmark clarity, not for headline ambition alone

## Sources Checked

Primary sources:
- GEM: A Native Graph-based Index for Multi-Vector Retrieval  
  https://arxiv.org/abs/2603.20336
- Multi-Vector Index Compression in Any Modality  
  https://arxiv.org/abs/2602.21202
- A Brief Comparison of Training-Free Multi-Vector Sequence Compression Methods  
  https://arxiv.org/abs/2603.22434
- LEMUR: Learned Multi-Vector Retrieval  
  https://arxiv.org/abs/2601.21853
- Spike Hijacking in Late-Interaction Retrieval  
  https://arxiv.org/abs/2604.05253

Repo context checked before choosing:
- [docs/architecture/extensibility_wall.md](architecture/extensibility_wall.md)
- [docs/architecture/gem_engine_integration.md](architecture/gem_engine_integration.md)
- [docs/python_sdk_charter.md](python_sdk_charter.md)
- [docs/python_sdk_roadmap.md](python_sdk_roadmap.md)
- [docs/benchmark_ladder.md](benchmark_ladder.md)

## Shortlist

### 1. GEM

Paper:
- **GEM: A Native Graph-based Index for Multi-Vector Retrieval**
- submitted `2026-03-20`

Short definition:
- a native graph-based stage-1 engine for multi-vector retrieval

Primary seam:
- stage-1 engine family

Why it fits:
- Kayak already isolates exact reranking as stage 2
- Kayak already has a GEM-family scaffold and integration note
- a faithful GEM reproduction would test whether Kayak can host a new native
  engine family without rewriting the exact reference path

### 2. Multi-Vector Index Compression in Any Modality

Paper:
- **Multi-Vector Index Compression in Any Modality**
- submitted `2026-02-24`

Short definition:
- constant-budget document-representation compression for late interaction

Primary seam:
- document-representation transform

Why it fits:
- Kayak already treats vectors/document, bytes/vector, and exact-reference
  recall as first-class benchmark outputs
- the paper maps cleanly to index-time transforms instead of a new serving
  contract

### 3. Training-Free Sequence Compression Comparison

Paper:
- **A Brief Comparison of Training-Free Multi-Vector Sequence Compression
  Methods**
- submitted `2026-03-23`

Short definition:
- compares training-free sequence compression choices such as pruning versus
  merging

Primary seam:
- document-representation transform plus benchmark contract

Why it fits:
- this is the narrowest recent paper family to reproduce honestly in Kayak
- it can reuse the current exact-search benchmark loop
- it provides a good first target for a paper-shaped benchmark surface

### 4. LEMUR

Paper:
- **LEMUR: Learned Multi-Vector Retrieval**
- submitted `2026-01-29`

Short definition:
- learned reduction from multi-vector search to latent single-vector ANN search

Primary seam:
- stage-1 engine family

Why it fits:
- it is a serious approximate retrieval paper with a clear stage-1 identity
- it is lower-priority than GEM because it adds a learned latent reduction and
  external ANN coupling on top of the engine work

### 5. Spike Hijacking

Paper:
- **Spike Hijacking in Late-Interaction Retrieval**
- submitted `2026-04-06`

Short definition:
- studies robustness and brittleness of hard MaxSim versus smoother pooling

Primary seam:
- stage-2 scoring or aggregation semantics

Why it fits:
- it is very recent and scientifically interesting
- it is not the best first systems target because the current repo is stronger
  at runtime and benchmark work than at training-dynamics reproduction

## First Implementation Decision

Decision:
- start with the **training-free sequence compression** family

Reason:
- it changes one seam cleanly
- it can be benchmarked against the current exact reference path without
  introducing a new native candidate engine immediately
- the repo already has transform runtime support for prefix pruning and token
  pooling, so the first missing piece is one paper-shaped comparison surface
  rather than a brand-new kernel family

What this first step should prove:
- Kayak can express training-free document-representation transforms through the
  same explicit transform seam
- Kayak can compare those transforms on one frozen benchmark surface with:
  - requested vector budget
  - realized vector count
  - exact-reference recall
  - judged quality
  - search latency
  - artifact bytes/document and bytes/vector

What this first step does not claim:
- that Kayak already reproduces the full paper result
- that token pooling is already the same as every token-merging method in the
  paper family
- that the best long-term target is no longer GEM

## Next Target After The First Compression Pass

If the compression benchmark surface is sound and useful, the next paper target
should be:

- GEM, as the flagship stage-1 engine reproduction

Reason:
- it is the strongest systems paper in the recent window
- the repo already has a GEM-family scaffold, so benchmark and artifact
  continuity can be preserved while the engine grows
