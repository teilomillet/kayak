# Recent Paper Targets

Status: working paper shortlist  
Date: `2026-05-01`

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
- Efficient Multivector Retrieval with Token-Aware Clustering and
  Hierarchical Indexing
  https://arxiv.org/pdf/2604.28142

Repo context checked before choosing:
- [docs/architecture/extensibility_wall.md](architecture/extensibility_wall.md)
- [docs/architecture/gem_engine_integration.md](architecture/gem_engine_integration.md)
- [docs/python_sdk_charter.md](python_sdk_charter.md)
- [docs/python_sdk_roadmap.md](python_sdk_roadmap.md)
- [docs/benchmark_ladder.md](benchmark_ladder.md)

## Shortlist

### 1. Tachiom

Paper:
- **Efficient Multivector Retrieval with Token-Aware Clustering and
  Hierarchical Indexing**
- submitted `2026-04-30`

Short definition:
- token-aware centroid allocation plus hierarchical multivector retrieval

Primary seam:
- stage-1 engine family plus compressed refine layout

Why it fits:
- Kayak already treats stage-1 candidate generation, exact stage-2 reranking,
  vector counts, and bytes/vector as first-class reporting axes
- the paper attacks the same late-interaction systems bottleneck Kayak is
  designed around: centroid quality, candidate gather cost, and residual
  scoring layout
- the algorithm depends on token identity, which is currently not a first-class
  aligned artifact in Kayak's low-level encoded-vector contracts, so a faithful
  implementation must add or import token ids explicitly

Implementation note checked on `2026-05-01`:
- the repo now has an additive first-gate probe in
  `python/kayak_bridge/tachiom_probe.py`
- the probe implements token-aware centroid allocation and exact centroid-scan
  candidate generation, followed by exact MaxSim reranking over the candidate
  window
- the candidate-generation and rerank path is now also available through a
  dim128 Mojo primitive in `kayak/search/tachiom_tac_dim128.mojo`
- a benchmark-only TAC profiler is available in
  `kayak/search/tachiom_tac_profile_dim128.mojo`, an experimental TAC+i8
  compressed rerank lane is available in
  `kayak/search/tachiom_tac_i8_dim128.mojo`, a residual-PQ refine lane is
  available in `kayak/search/tachiom_tac_pq_dim128.mojo`, and a native
  query-time HNSW centroid traversal is available in
  `kayak/search/tachiom_tac_hnsw_dim128.mojo`
- Kayak's Python late-document/index contracts now carry optional aligned
  document token ids, and the ColBERT encoder preserves them when its
  checkpoint exposes `doc_tokenizer`
- task JSON builders can now emit aligned ColBERT document token ids with
  `--include-document-token-ids`, and `python/scripts/bench_tachiom_task.py`
  refuses vector-only judged tasks for Tachiom runs
- `python/kayak_bridge/tachiom_full.py` composes the paper-shaped TAC + HNSW
  candidate path with residual-PQ rerank for a judged-task reproduction gate
- it intentionally does **not** claim the full paper result yet: HNSW exists as
  a first in-tree reference, but there is no paper-scale PQ/HNSW evaluation and
  the measured HNSW frontier does not yet match the paper; the i8 lane is a
  compressed rerank probe, while the residual-PQ lane is the first
  paper-shaped refine implementation
- smoke evidence is recorded in
  [docs/traces/2026-05-01_tachiom_tac_probe.md](traces/2026-05-01_tachiom_tac_probe.md)
- current interpretation: Mojo TAC is a useful high-recall native candidate
  lane on token-structured synthetic shapes; profiling showed exact candidate
  rerank, not centroid scan, dominates the tested full-recall operating points
- the TAC+i8 compressed rerank probe reduces the large-shape full-recall
  footprint to `0.304x` exact-vector bytes while keeping `82.29` QPS versus
  `34.43` QPS for exact NumPy, but the moderate shape still stops at `0.9875`
  final recall
- the residual-PQ smoke sweep reaches exact final recall with `128` PQ
  subspaces at `0.509x` exact-vector bytes, and reaches `0.975` final recall
  with `64` subspaces at `0.384x` exact-vector bytes; both are lower-QPS than
  TAC+i8 on the smoke shape
- sampled moderate residual-PQ runs (`8192` training tokens) keep candidate
  recall at `1.0`; `64` subspaces gives `0.95` final recall at `0.208x`
  exact-vector bytes and `142.67` QPS, while `128` subspaces gives `0.9875`
  final recall at `0.333x` exact-vector bytes and `69.01` QPS
- the current residual-PQ lane therefore improves memory and paper-shape
  fidelity, but it is not yet a better measured latency/quality frontier than
  TAC+i8 on the moderate workload
- Candidates Pruning is now implemented in Python and native TAC/PQ paths; with
  `alpha=0.4`, it reduces the moderate PQ128 candidate window from `96` to a
  mean `77.5`, preserves `0.9875` final recall, and raises QPS from about
  `69.01` to `84.48`
- an HNSW-style centroid graph is now implemented with Python graph
  construction, diversified neighbor pruning, and native dim128 query
  traversal; it reaches full recall and slightly beats exact centroid scan on
  the large synthetic shape (`97.46` versus `95.26` QPS), but graph build time
  is still Python-bound and much slower
- the cached LEMB NarrativeQA judged task was checked and rejected for Tachiom
  benchmarking because it lacks aligned document token ids; this is a faithful
  blocker, not a search result
- a new LEMB NarrativeQA q8 artifact was rebuilt with aligned document token
  ids (`355` docs, `63900` doc vectors, `8` queries, `32` mean query vectors);
  on that small judged gate, TAC+HNSW+exact rerank and TAC+HNSW+PQ64 both match
  exact nDCG@10 (`0.375`) but only recover about `0.81` / `0.80` of exact
  top-10 rankings, and exact Kayak is much faster (`697` QPS versus `73` and
  `32` QPS)
- MS MARCO dev.small token-id gates are now built from official `ir_datasets`
  files; the larger bounded gate has `5135` selected docs, `373889` doc
  vectors, `128` queries, MRR@10, zero missing selected positives, and uses
  paper TAC allocation knobs (`mu=128`, `tau=256`, `epsilon=4`, `theta=39`)
- on that larger MS MARCO gate, `8192` centroids do not match exact MRR@10
  (`0.9570` versus `0.9961`), and exact-rerank diagnostics show the loss is
  in centroid-only gather rather than residual PQ
- raising TAC to `32768` centroids closes the bounded quality gate:
  exact-rerank, exact-centroid+PQ32, and HNSW+PQ32 all match exact MRR@10
  (`0.99609375`), with the HNSW+PQ32 run using PQ32, HNSW `M=32`,
  `ef_construction=1500`, `ef_search=180`, `k_c=120`, `k_d=1000`, and
  `alpha=0.35`
- the measured HNSW+PQ32 quality-matched run is not a speed reproduction:
  exact packed search is about `156.7` QPS on this small slice, while the
  Python reference HNSW+PQ32 run is about `10.35` QPS and spends `444s` in
  build
- this MS MARCO result is a reproduction gate, not the paper result: the paper
  reports MS MARCO-v1 at `8.8M` passages, `598M` token vectors, `6980`
  dev.small queries, and about `4M` centroids; the current JSON/full-float path
  cannot materialize that corpus on the verified local disk budget, so the next
  full-reproduction requirement is a compressed or streaming paper-scale build
  path
- that compressed materialization path now exists: the MS MARCO ColBERT
  snapshot writer streams `float16` document vectors, `uint32` token ids,
  document offsets, doc ids, and query sidecars into resumable binary shards;
  the paper-scale document payload estimate is `155550734592` bytes
  (`153088000000` vector bytes plus `2392000000` token-id bytes), which fits
  the verified local disk budget
- the sharded materializer has been smoke-tested on official MS MARCO files and
  resume-tested by extending a snapshot from `128` to `160` documents
- the snapshot-to-Tachiom bridge now exists and was smoke-tested on the
  MS MARCO binary snapshot: it builds TAC+HNSW+PQ directly from `float16`
  vector shards, `uint32` token-id shards, offsets, and query sidecars without
  task JSON; the smoke loaded `128` documents and `8752` document vectors and
  produced a `593544`-byte HNSW+PQ index
- that bridge still uses the current in-memory TAC reference after loading the
  selected shard rows, so it is a selected-slice benchmark bridge rather than
  the final paper-scale index builder
- the streaming TAC/PQ builder now connects the binary snapshot format directly
  to on-disk TAC/PQ artifact construction without task JSON, Python vector
  lists, or a full in-RAM document-token matrix; the smoke artifact has `160`
  documents, `10902` document vectors, `256` centroids, `3581` postings,
  `620422` query-time index bytes, and `882054` build bytes including retained
  centroid samples
- the streaming artifact reader now searches that on-disk artifact through
  memmaps; on the same `160`-doc prefix smoke it ran at `171.23` QPS and
  reached `0.75` candidate/final recall@10 versus exact, while judged MRR
  stayed `0.0` because the prefix does not force qrels positives into the
  selected documents
- a persisted centroid HNSW sidecar now exists for streaming artifacts; on the
  same smoke it built a `256`-centroid, `4408`-edge graph in `0.242s`, added
  `28192` graph bytes, and searched at `38.60` QPS with the same `0.75`
  candidate/final recall@10 versus exact
- bounded streaming snapshots can now include required positives for selected
  judged queries via `--include-query-positives`; on an MS MARCO dev.small
  judged-positive slice (`264` documents, `18846` document vectors, `8`
  queries, `8` required positives), streaming TAC+PQ32 with `4096` centroids
  matches exact MRR@10 (`1.0` versus `1.0`) while keeping final top-10 overlap
  at `0.8875`
- the persisted HNSW sidecar also matches exact MRR@10 on that `4096`-centroid
  judged-positive slice, but it is slower than exact centroid scan there
  (`14.68` QPS versus `40.17` QPS), so it is validated as a graph artifact
  path rather than as paper-throughput evidence
- the larger `docs1000_q32` judged-quality gate has now been reproduced
  through the streaming path rather than task JSON: the snapshot has `1036`
  documents, `74804` document vectors, `32` queries, and `36` required
  positives; streaming TAC+PQ32 with `8192` centroids matches exact MRR@10
  (`1.0` versus `1.0`) at `31.25` QPS, and streaming HNSW+PQ32 also matches
  exact MRR@10 at `13.30` QPS
- retrying that streaming gate with `32768` centroids keeps MRR@10 matched and
  improves exact top-10 overlap from `0.846875` to `0.921875`, but query speed
  drops to `11.46` QPS and query-time index bytes rise from `7.62M` to
  `20.67M`
- `python/scripts/run_tachiom_streaming_scale_gate.py` now supports repeatable
  judged-positive streaming scale gates; on a new `docs1500_q48` run (`1553`
  documents, `114393` vectors, `48` queries, `53` required positives),
  streaming TAC+PQ32 with `32768` centroids again matches exact MRR@10
  (`1.0` versus `1.0`) with `0.9104` final top-10 overlap and `11.37` QPS,
  while HNSW+PQ32 also matches MRR with `0.9042` final overlap and `9.56` QPS
- the streaming exact-centroid query path was then profiled and optimized on
  that same artifact: top-k centroid selection moved from full sort to bounded
  `argpartition`, selected posting accumulation now uses one `np.maximum.at`
  per query token, and exact-centroid TAC+PQ32 improved from `11.37` to
  `37.65` QPS with unchanged MRR and exact top-10 overlap
- the persisted HNSW query path was then profiled separately and optimized by
  batching neighbor dot products in graph traversal; HNSW+PQ32 improved from
  `10.62` to `15.77` QPS after the shared posting pass, still with unchanged
  MRR and exact top-10 overlap
- the Python HNSW graph builder also now batches neighbor score computation;
  rebuilding the same `32768`-centroid graph reduced build time from `38.61s`
  to `35.63s`, but this remains a Python reference builder rather than a
  kANNolo-class paper-scale builder
- a native streaming TAC+PQ engine (`streaming_tac_pq_mojo`) now prepares the
  streaming artifact into the dim128 residual-PQ Mojo primitive; on the same
  `docs1500_q48_c32768` slice, it produced exactly the same final rows as the
  Python streaming TAC+PQ path and reached `73.06` QPS with MRR@10 `1.0`,
  compared with `47.26` QPS for exact MaxSim on the same snapshot
- a native streaming HNSW+PQ engine (`streaming_tac_hnsw_pq_mojo`) now uses the
  persisted centroid graph plus sparse residual-PQ rerank over candidate
  document tokens; it reached `125.52` QPS on `docs1500_q48_c32768` with final
  recall@10 vs exact `0.90417`, and `69.01` QPS on `docs10000_q128_c32768`
  with final recall@10 vs exact `0.88672`
- an address-backed native streaming HNSW+PQ variant
  (`streaming_tac_hnsw_pq_mojo_address`) now avoids Python-list materialization
  of the memmap-backed index arrays; on the 10k row it prepared in `0.016s`
  versus `2.00s` for the List-backed native prepare, but query speed was lower
  at `42.24` QPS, so it is a materialization-safe path rather than the default
  fastest query path
- larger bounded native streaming gates also hold: `docs2500_q64` reached
  `73.48` QPS versus `27.77` exact MaxSim QPS, `docs5000_q96` reached
  `58.05` QPS versus `13.88` exact MaxSim QPS, and `docs10000_q128` reached
  `55.84` QPS versus `6.90` exact MaxSim QPS; all use `32768` centroids and
  keep judged MRR matched or slightly above exact, while exact top-10 overlap
  remains approximate at about `0.88` to `0.89`
- the `docs10000_q128` run also exposed CPU ColBERT snapshot materialization as
  a practical wall-clock bottleneck, so the scale runner now emits per-stage
  progress and timing instead of staying quiet through long builds
- USL batch-cap sweep tooling now exists for the native streaming HNSW+PQ
  reader; on the bounded `docs1500_q48` and `docs10000_q128` artifacts the
  measured same-shape batch-cap gain was under `1%`, and both USL fits had
  negative `r_squared`, so batching is not the current material bottleneck
- this is still not the paper result: large token partitions use deterministic
  sampled centroids when bounded Python k-means limits would be exceeded, the
  HNSW graph builder is still the Python reference rather than a
  kANNolo-class/native paper-scale builder, the native HNSW+PQ path shows a
  measured speed/quality tradeoff rather than exact-centroid equivalence, and
  no full MS MARCO/LoTTE paper-scale judged run has completed
- the next sound Tachiom gate is therefore a judged streaming snapshot slice
  with forced positives, larger-centroid HNSW sidecar measurements,
  residual-PQ quality/speed tuning, or a decision that paper-throughput is out
  of scope for the current Python/kANNolo-free path before claiming paper
  reproduction

### 2. GEM

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

### 3. Multi-Vector Index Compression in Any Modality

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

### 4. Training-Free Sequence Compression Comparison

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

### 5. LEMUR

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

### 6. Spike Hijacking

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
