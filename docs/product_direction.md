# Product Direction

Status: narrowed direction note
Date: `2026-04-25`

This note narrows Kayak's product direction after comparing the current repo
scope with the broader managed-RAG category.

The goal is to define what the project should optimize for next, and what it
should deliberately leave outside the mainline.

## Decision

Kayak should optimize for:

**the late-interaction search layer**

not:

- a full RAG pipeline
- a document-intelligence platform
- an OCR, parsing, or extraction system
- a generic vector database
- a thin client over a managed retrieval API

Reason:

- the repo already has strong local evidence and implementation surface around
  late-interaction objects, exact MaxSim scoring, search plans, snapshots,
  stage-aware benchmarks, and hosted retrieval contracts
- the repo does not currently own the production document-ingestion problems
  that define a full RAG platform: OCR, layout parsing, table extraction,
  chunk reconstruction, and raw-document lifecycle
- narrowing the boundary lets optimization work compound instead of spreading
  across unrelated pipeline problems

## Where Kayak Starts

Kayak's canonical starting point is one of:

1. encoded late-interaction query and document representations
2. plain text plus a caller-selected encoder that emits token-level vectors
3. a materialized search slice loaded from an existing storage system

Kayak should not claim that raw enterprise documents are its canonical input.

Reason:

- the current architecture treats the encoder boundary as external
- the hosted service contracts already preserve model identity, vector
  dimensions, vector counts, search plans, and snapshot identity explicitly
- accepting raw documents as the default product story would imply ownership of
  parsing and extraction quality that the repo does not verify

## Where Kayak Ends

Kayak's canonical output is retrieval evidence:

- document ids
- scores
- plan metadata
- stage profiles
- explain/debug data

Kayak should not make answer generation the core product boundary.

Reason:

- the repo is strongest where retrieval behavior can be inspected, measured,
  and compared against exact references
- generation quality depends on prompt, model, context assembly, policy, and
  application workflow choices outside the current engine contract

## Optimization Lanes

The next optimization work should fit one of these lanes.

The current measurement surface and latest optimization evidence are tracked in
[docs/search_layer_optimization_scorecard.md](search_layer_optimization_scorecard.md).

### GPU Readiness Gate

Before making GPU the main implementation focus, Kayak should close or clearly
label the current CPU Pareto gap against FastPlaid.

Verified evidence on `2026-04-25`:

- Kayak dominates FastPlaid on the `cpu_matrix_v2_smoke` small and medium CPU
  shapes in both raw and normalized modes.
- Kayak does not dominate FastPlaid on the large
  `2048`-document, `32`-document-vector, `96`-query-vector CPU smoke shape.
  The exact-recall Kayak point stores much more bytes, and the normalized row
  is slightly slower. The pruned Kayak points are faster but lose too much
  exact-reference recall.

Reason:

- GPU work should accelerate a search plan whose CPU tradeoffs are already
  understood. Otherwise a GPU kernel can hide an unresolved candidate-recall or
  storage-efficiency problem.

The next CPU work is therefore:

- improve the large-shape pruned candidate path
- add a compressed-vector or score-proxy lane with measured bytes
- keep `PlaidApproxConfig` as the explicit public parameter for users who want
  the approximation tradeoff

### Lane 1: Exact Search Throughput

Optimize the exact late-interaction reference path.

Examples:

- reduce Python-to-Mojo conversion overhead
- improve prepared same-snapshot reuse
- improve batch search over one loaded index
- improve layout-specialized scoring kernels
- maintain a FastPlaid speed track where Kayak's Mojo exact path and opt-in
  Mojo approximation path are compared against FastPlaid's Rust/PLAID-style path
  on the same explicit shapes
- expose approximate PLAID-style search as an explicit opt-in parameter rather
  than silently replacing exact MaxSim

Required evidence:

- exact score parity against the NumPy reference or a simpler Mojo reference
- latency and vector-count reporting on fixed query/document shapes
- for FastPlaid comparisons: recall against Kayak exact, index build/update
  time, memory or index bytes, device, and FastPlaid compression controls
- for Kayak approximate lanes: the candidate budget, centroid/posting controls,
  and exact-reference recall must appear in the report

Reason:

- the exact path is both a usable product path and the correctness anchor for
  every approximate plan
- FastPlaid is the closest current open-source speed reference for this lane,
  so Kayak should compete against it directly while preserving exact-reference
  accountability
- approximation is a user-facing tradeoff, not an implementation detail; callers
  should be able to choose speed only when the measured recall/cost profile is
  acceptable

### Lane 2: Candidate Recall Per Unit Cost

Optimize stage-1 candidate generation while keeping stage-2 exact reranking
explicit.

Examples:

- improve `document_proxy`
- improve centroid or graph-like native candidate families
- tune candidate windows and vector budgets
- make filter-aware candidate generation cheaper and more faithful

Required evidence:

- candidate recall against exact full-scan references
- final quality after exact rerank
- candidate count, query vector count, and document vector count

Reason:

- a cheaper stage 1 only matters if it actually reaches the documents that
  exact reranking needs

### Lane 3: Storage And Vector Budget Efficiency

Optimize bytes per vector, vectors per document, and load-time behavior.

Examples:

- compressed vector payloads
- document-representation transforms
- training-free token pooling
- storage layouts that reduce reload and memory pressure

Required evidence:

- bytes per vector
- bytes per document
- quality and latency before/after
- vector-count distributions, not only averages

Reason:

- late interaction's product cost is controlled by vector count and storage
  layout as much as by raw scorer speed

### Lane 4: Operational Search Service Trust

Optimize Kayak Engine as a deployable retrieval service.

Examples:

- snapshot publication safety
- prepared runtime lifecycle
- metrics and explain surfaces
- reclaim and retention workflows
- explicit overload and backpressure behavior

Required evidence:

- integration tests around lifecycle behavior
- operator-visible metrics
- failure-mode documentation

Reason:

- teams adopting retrieval infrastructure buy operational confidence, not only
  a kernel benchmark

### Lane 5: Integration Boundaries

Make it easy to connect Kayak to external parsers, encoders, vector databases,
and application stacks without making those systems part of Kayak's identity.

Examples:

- clearer encoder adapters
- database handoff examples
- artifact import/export
- plain-text ingestion convenience APIs

Required evidence:

- runnable examples
- explicit data-shape and trust-boundary docs
- no hidden model, layout, or backend switching

Reason:

- integration smoothness helps adoption, but the core product should remain
  late-interaction search rather than a general RAG pipeline.

## Explicit Non-Goals

Do not optimize mainline work for:

- OCR quality
- PDF layout recovery
- table extraction
- handwritten annotation support
- LLM answer generation
- broad workflow automation for legal, finance, procurement, or construction
  users

Those can be integration partners, examples, or future product decisions, but
they should not consume the core search-engine roadmap without a separate
decision note.

## Public Story

The short public story should be:

> Kayak is the late-interaction search layer for teams that need retrieval they
> can inspect, measure, deploy, and tune.

Supporting points:

- exact MaxSim is the reference path
- candidate stages are explicit, not hidden
- vector counts are cost and quality inputs
- layouts and backends are visible
- existing databases and external encoders can stay in the system
- raw document parsing and extraction remain outside Kayak's main boundary

This story is narrower than "RAG made easy" and more accurate to the current
repo.
