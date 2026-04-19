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

Implementation note checked on `2026-04-19`:
- the repo already has a paper-shaped in-tree `gem_graph` family with exact
  transport, adaptive cutoff, and shortcut injection on the active builder path
- the repo now also has a held-out synthetic GEM ablation surface that trains
  adaptive and shortcut-aware graph variants on disjoint judged queries before
  evaluation, so supervised GEM features can be benchmarked without reusing the
  same query labels for both build-time supervision and held-out measurement
- the held-out adaptive diagnostics now show a concrete current weakness:
  on both the tiny smoke slice and the first real synthetic profile, the
  current adaptive supervision labels collapse to width `1`, so the next GEM
  question is not just tree quality but whether the adaptive label definition is
  strong enough for conjunction-heavy workloads
- the repo now keeps the paper-faithful label policy explicit as
  `first_relevant_cluster_rank` and also exposes an experimental
  `relevant_cluster_coverage` policy; that experimental policy fixes the smoke
  regression but does not rescue the first real synthetic slice, so traversal
  quality remains a separate GEM problem
- the held-out artifact now distinguishes profile hit, entry-point coverage, and
  gated reachability; on the first real synthetic profile, baseline and coverage
  adaptive both keep relevant positives reachable yet still miss them at small
  candidate windows, which narrows the remaining issue to query-time search
  order or candidate budget rather than graph disconnection
- a dedicated beam probe on that same hard slice shows beam width is a real
  lever but not a monotonic fix, so the next sound GEM question is how
  query-time budget and ordering interact rather than whether one more static
  pruning tweak will solve the slice
- a fixed-beam candidate-window probe now makes that budget question concrete:
  on the first real synthetic slice, baseline recall stays at `0.0` through
  `candidate_k=16`, rises to `0.5` at `32`, and reaches `1.0` at `64`, while
  both adaptive variants improve more slowly and entry rate stays `0.0`
  throughout; that points to deeper traversal budget rather than better seeded
  entry points as the current recovery mechanism
- a dedicated cluster-gate probe then shows widening
  `cluster_top_k_per_query_token` is not a clean rescue:
  baseline improves only modestly at `candidate_k=32` when widened to `4`, but
  the default gate `2` remains best at `candidate_k=64`, and larger gates add
  cost without monotonic gains
- a representative-depth probe rules out the simplest entry-depth story:
  even top-`8` per-cluster representatives never hit the positives on the first
  real synthetic slice under the tested cluster gates, so a simple multi-entry
  seeding primitive is unlikely to be the highest-value next GEM change
- a construction-density probe then constrains the baseline graph itself:
  low `degree_limit` collapses reachability, higher
  `construction_neighbor_count` is actively harmful when degree is starved, and
  denser-than-default graphs do not beat the default `neighbors=6`,
  `degree=8` setting at high budget on the first real synthetic slice
- new hop diagnostics then make the remaining failure mode more explicit:
  on that same slice, baseline positives are reachable but still sit about
  `4.8` gated graph hops away on average, while adaptive variants push them to
  about `8-10` hops even when reachability remains high
- an entry-hop probe then shows that better seeded entries help only modestly:
  representative-based seeds reduce baseline mean hop count only to about `4.2`
  at depth `8`, and adaptive variants remain much deeper
- re-running the construction surface with those hop diagnostics shows that
  shorter path length alone is not enough: higher-degree graphs can shorten
  the path to positives materially without beating the default high-budget
  baseline recall
- a frontier-policy probe then constrains the search-side story:
  replacing the current per-entry local queues with a pure global best-first
  frontier lowers search cost but hurts the strongest current baseline badly
  (`1.0` to `0.5` recall at `candidate_k=64`), so root diversity appears to be
  useful rather than accidental
- a follow-up hybrid frontier-policy probe then narrows that story further:
  preserving one queue per entry root while expanding the globally best queue
  head recovers part of the lost baseline medium-budget recall
  (`0.417` versus `0.333` for pure global at `candidate_k=32`) and materially
  helps the coverage-adaptive variant (`0.5` versus `0.167` at
  `candidate_k=32`), but it still fails to recover the strongest baseline
  high-budget point (`0.5` versus `1.0` local at `candidate_k=64`); that
  suggests root-capacity preservation alone is insufficient and the remaining
  query-time question is whether the useful ingredient is more explicit
  root-fairness rather than just more global ordering
- a follow-up root-fair frontier-policy probe then closes most of that
  scheduling question:
  a one-expansion fair round recovers the local baseline almost exactly, and a
  looser quota-`2` round still collapses back to local behavior on the tested
  slice; that means the strong baseline gain really does come from root-fair
  scheduling, but simple quota tuning does not yield a better middle point, so
  remaining GEM gains likely need either a more stateful diversity primitive or
  a stronger structural graph change rather than another simple queue-ordering
  scalar
- a follow-up implementation step then promoted that frontier choice into an
  explicit graph-family primitive shared by planning, live execution, service
  JSON, and the held-out probe itself; the verified result is recorded in
  [docs/traces/2026-04-19_gem_frontier_policy_primitive.md](traces/2026-04-19_gem_frontier_policy_primitive.md)
  and keeps default serving behavior at `local_per_entry` while making
  `graph_frontier_policy_kind` reproducible and inspectable across the main
  research surface
- a shortcut probe then rules out the easy bridge-injection story:
  the plain shortcut variant injects `0` shortcut edges and matches baseline,
  while `adaptive_cutoff_shortcuts` injects only `2` shortcut edges and still
  matches paper-faithful adaptive exactly on the first real synthetic slice
- a follow-up shortcut-budget probe rules out the easy scalar rescue:
  widening `shortcut_candidate_k` from `6` to `64` changes neither shortcut
  edge count nor recall for either shortcut-enabled variant on that slice
- a follow-up implementation change then allowed shortcut insertion to rewire
  saturated degree lists rather than only append into spare slots, and the
  held-out shortcut and shortcut-budget probes still stayed unchanged; that
  falsifies the simpler "degree saturation is blocking useful shortcuts"
  explanation too
- the held-out query-time probe now persists and validates
  `shortcut_candidate_k` explicitly, so future sweeps cannot silently reuse
  stale GEM caches with collapsed shortcut-budget metadata
- the next GEM question is therefore no longer "can Kayak host GEM at all?"
- the next GEM question is whether a stronger structural graph primitive
  or a more diversity-aware query-time frontier policy can improve the measured
  frontier enough to justify GEM as a preferred stage-1 family
- the sound wrap point for the current GEM frontier cycle is therefore:
  keep the explicit frontier-policy primitive, keep `local_per_entry` as the
  default until new evidence beats it, and stop spending more time on simple
  queue-ordering scalars before testing a stronger structural or stateful
  diversity change
- a fresh paper-faithfulness gap check on `2026-04-20` then made the current
  practical limit explicit on the shared BrowseComp-plus gold surface:
  `document_proxy` still reaches full recall earlier and faster, while
  `gem_graph` visits about `89.25` of `90` documents at its first full-recall
  point; see
  [docs/traces/2026-04-20_gem_paper_faithfulness_gap_check.md](traces/2026-04-20_gem_paper_faithfulness_gap_check.md)
  for the current boundary between mechanical correctness, paper-shaped
  alignment, and still-missing pruning advantage

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

Implementation note checked on `2026-04-19`:
- the public reference code exposes a mixed family surface
- hierarchical pooling is compatible with stored-document transform execution
- sequence resizing, memory tokens, and attention-guided clustering are
  encoder-bound in the reference implementation
- the sound Kayak primitive is therefore explicit family classification plus
  transform lowering only for the stored-representation-executable subset

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
- the current in-tree GEM implementation shifts the next decision from
  scaffolding to pruning-quality and frontier-quality work rather than a first
  artifact/planning pass
