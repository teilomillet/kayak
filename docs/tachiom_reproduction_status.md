# Tachiom Reproduction Status

Date: `2026-05-03`

Paper:
- **Efficient Multivector Retrieval with Token-Aware Clustering and
  Hierarchical Indexing**
- <https://arxiv.org/pdf/2604.28142>

## Verdict

Status: **paper-shaped implementation exists; bounded MS MARCO quality gates
pass; full paper-scale throughput is not reproduced.**

This document is the canonical claim boundary for Kayak's Tachiom work. It
separates:

- implemented algorithmic components
- bounded local reproduction evidence
- paper-scale results that are still unverified
- exact gates that would upgrade the claim

Reason:
- the codebase now contains enough Tachiom machinery that loose language like
  "we reproduced the paper" would be misleading
- the current strongest evidence is real and useful, but it is not the same
  scale, implementation substrate, or throughput setting as the paper

## Sources Checked

Paper source checked on `2026-05-02`:
- the paper evaluates MS MARCO-v1 with `8.8M` passages, `598M` token vectors,
  and `6980` dev.small queries
- the paper evaluates LoTTE-pooled with `2.4M` passages, `266M` token vectors,
  and `2931` search/dev queries
- the paper uses ColBERTv2, reports MRR@10 for MS MARCO-v1 and Success@5 for
  LoTTE, and reports an implementation written in Rust on top of kANNolo
- the paper uses TAC thresholds `mu=128`, `tau=256`, active-token floor
  `epsilon=4`, `theta=39`, PQ32 with 8-bit codes, and `10` k-means iterations
- the paper's retrieval setup uses about `2M` centroids for LoTTE and about
  `4M` centroids for MS MARCO-v1, HNSW `M=32`, `ef_construction=1500`, and a
  grid over `k_c`, `k_d`, and candidate-pruning `alpha`

Local evidence source:
- [docs/traces/2026-05-01_tachiom_tac_probe.md](traces/2026-05-01_tachiom_tac_probe.md)
- [docs/traces/2026-05-02_tachiom_pruning_sweep.md](traces/2026-05-02_tachiom_pruning_sweep.md)
- [docs/traces/2026-05-02_tachiom_centroid_ladder.md](traces/2026-05-02_tachiom_centroid_ladder.md)
- [docs/traces/2026-05-02_tachiom_query_policy_sweep.md](traces/2026-05-02_tachiom_query_policy_sweep.md)
- [docs/traces/2026-05-02_tachiom_hnsw_heap_frontier.md](traces/2026-05-02_tachiom_hnsw_heap_frontier.md)
- [docs/traces/2026-05-03_tachiom_rerank_hot_loop.md](traces/2026-05-03_tachiom_rerank_hot_loop.md)
- [docs/traces/2026-05-03_tachiom_hnsw_visited_table.md](traces/2026-05-03_tachiom_hnsw_visited_table.md)
- [docs/benchmark_ladder.md](benchmark_ladder.md)
- [docs/recent_paper_targets.md](recent_paper_targets.md)

## Status Key

- `Implemented`
  - the repo has code for the component and focused tests cover its local
    contract
- `Bounded reproduction gate`
  - the component or result has been measured on a stated local slice with
    explicit vector counts
- `Partial`
  - the shape exists, but an important paper equivalence condition is missing
- `Not reproduced`
  - the current repo does not yet provide evidence for the paper claim

## Paper vs Kayak

| Paper item | Paper target | Kayak implementation | Current evidence | Status |
| --- | --- | --- | --- | --- |
| Token-aware clustering | TAC allocates centroids by token frequency and variance with tail handling, damped scoring, bounds, and budget reconciliation | `python/kayak_bridge/tachiom_allocation.py`, `tachiom_clustering.py`, `tachiom_index.py` | unit tests and bounded MS MARCO gates use `mu=128`, `tau=256`, `epsilon=4`, `theta=39` | `Implemented` |
| Aligned token ids | TAC depends on document token identity aligned to document vectors | ColBERT encoder path and task/snapshot builders preserve aligned document token ids | vector-only cached LEMB artifact was rejected; token-id-bearing artifacts were rebuilt | `Implemented` |
| Candidate pruning and query policy | paper reports a grid over `k_c`, `k_d`, and candidate-pruning `alpha` | shared candidate ranking path supports `candidate_pruning_alpha`; streaming benchmark now supports query-time overrides for `centroids_per_query_vector`, HNSW `ef_search`, and pruning | alpha `0.05` roughly doubled docs10000 bounded QPS while preserving judged MRR, but exact top-10 overlap fell to `0.5625`; larger-centroid co-sweeps recovered exact-overlap floors only at slower QPS | `Bounded reproduction gate` |
| HNSW centroid traversal | graph over centroids, paper settings include `M=32`, `ef_construction=1500` | Python graph builder plus native dim128 query traversal and streaming sidecar | bounded gates use persisted HNSW sidecars; native HNSW+PQ reaches `125.52` QPS on `docs1500_q48_c32768` and `69.01` QPS on `docs10000_q128_c32768` | `Partial` |
| Residual-PQ refine | normalized residual compression, PQ32, 8-bit codes, optimized layout for MaxSim | Python residual-PQ reference and dim128 Mojo residual-PQ paths | PQ32 bounded gates exist; native sparse HNSW+PQ reranks candidate-window tokens | `Partial` |
| Streaming/materialized index | paper-scale implementation avoids full dense token scoring at query time | binary snapshot writer, streaming TAC/PQ builder, memmap reader, native list-backed and address-backed engines | full MS MARCO document payload estimate is about `155.55GB` with f16 vectors/u32 token ids; bounded streaming artifacts are built and searched | `Partial` |
| MS MARCO judged quality | MS MARCO-v1, `8.8M` passages, `598M` vectors, `6980` queries, MRR@10 | bounded selected-positive MS MARCO slices | `docs5000_q128` JSON gate reaches exact MRR@10 at `32768` centroids; streaming gates keep judged MRR matched or slightly above exact on bounded slices | `Bounded reproduction gate` |
| LoTTE judged quality | LoTTE-pooled, `2.4M` passages, `266M` vectors, `2931` queries, Success@5 | only a small LEMB/NarrativeQA token-id gate exists, not LoTTE | no LoTTE-pooled paper-dataset run | `Not reproduced` |
| Corpus scale | millions of passages and hundreds of millions of token vectors | largest measured bounded streaming row is `10135` documents and `739372` document vectors | no full MS MARCO or LoTTE corpus run | `Not reproduced` |
| Centroid scale | about `4M` centroids for MS MARCO-v1, about `2M` for LoTTE | bounded gates now reach `262144` centroids on the docs10000 snapshot | `262144` centroids materialized locally, but fixed-policy exact top-10 overlap fell to `0.8109374999999999`; no `1M`, `2M`, or `4M` local centroid gate has completed | `Partial` |
| Throughput scale | paper reports Tachiom average query times of `10-15ms` at MS MARCO cutoffs and speedups up to `9.8x` over baselines | native HNSW+PQ reaches comparable per-query milliseconds only on much smaller bounded slices | local QPS rows are not comparable to paper throughput because corpus and centroid scale differ by orders of magnitude | `Not reproduced` |
| Graph construction | Rust/kANNolo implementation, 64-thread clustering, single-core retrieval experiments | Python HNSW graph builder; native query traversal exists | Python graph build dominates or remains a blocker; no native/kANNolo-class builder | `Partial` |
| Hardware/runtime parity | Intel Xeon Silver 4314, 64 threads; Rust/kANNolo; retrieval sequential on one core | current local CPU/Python/Mojo environment | no hardware-parity run; no paper implementation replay | `Not reproduced` |

## Strongest Verified Local Claims

These claims are currently justified:

- Kayak has a paper-shaped TAC + HNSW + residual-PQ implementation path.
- Kayak can reject unfaithful vector-only Tachiom artifacts when aligned
  document token ids are missing.
- Bounded MS MARCO selected-positive gates can match exact MRR@10 when the
  centroid budget is high enough.
- Native streaming TAC+PQ and HNSW+PQ paths beat exact MaxSim on the measured
  bounded slices as document-vector count grows.
- Native HNSW+PQ is the fastest bounded query path measured so far, but it is
  approximate relative to exact top-10 rankings.
- USL batch-cap tuning is instrumented and measured; it did not reveal a
  material batching bottleneck on the current native HNSW+PQ slices.
- Native HNSW+PQ internal profiling is instrumented and measured on the same
  bounded slices; the 1.5k slice is HNSW-traversal dominated, the 10k `32768`
  centroid row is rerank-scoring dominated, and the 10k `262144` centroid
  stress row is still split between HNSW traversal and residual-PQ rerank
  scoring.
- Query-time candidate-pruning sweeps are instrumented and measured; aggressive
  alpha values can improve bounded throughput while preserving judged MRR, but
  exact top-10 overlap must be treated as a separate quality constraint.
- A centroid-count ladder is instrumented and measured up to `262144`
  centroids on the docs10000 bounded snapshot; larger centroid counts improved
  throughput under fixed `k_c=120` and `ef_search=64`, but exact top-10 overlap
  fell rather than improved.
- A larger-centroid query-policy co-sweep is instrumented and measured; it
  recovered the `32768` baseline final-recall floor on `131072` and `262144`
  centroid artifacts, but the quality-preserving rows were slower than the
  fixed-policy `32768` baseline.
- Native HNSW+PQ now uses heap-backed HNSW layer frontiers and skips the final
  retained-centroid sort when `ef == k_c`; this improves the measured `262144`
  quality-preserving row from `24.116374186902767` QPS to
  `72.61081722938026` QPS, while preserving MRR@10 and exact-overlap gates.
- A small rerank hot-loop cleanup was measured with controlled repeated A/B on
  the `32768` and `262144` focused rows; it is kept as a marginal local
  optimization, not as a new paper-reproduction result.
- Native HNSW+PQ now starts its sparse visited table at `ef * 32 + 16`
  instead of `ef * 64 + 16`; the current fastest `32768` row and `262144`
  stress row improved while preserving the same MRR@10 and exact-overlap gates.
  The `131072` eligible row was slightly slower, and the `ef * 16 + 16`
  variant was measured and rejected.
- Candidate pruning now uses a pruning-aware top-k helper that computes the
  final-rank cutoff first and only heap-ranks positions that can survive the
  pruning threshold. This improved the current `32768`, `131072`, and `262144`
  quality-preserving rows while preserving the same quality gates.

Current strongest bounded native streaming rows:

| Slice | Docs | Doc vectors | Queries | Query vectors | Centroids | Engine | MRR@10 | Exact MRR@10 | Final recall@10 vs exact | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `docs1500_q48_c32768` | `1553` | `114393` | `48` | `1536` | `32768` | native HNSW+PQ | `1.0` | `1.0` | `0.9041666666666663` | `125.52333181000189` |
| `docs2500_q64_c32768` | `2570` | `188537` | `64` | `2048` | `32768` | native TAC+PQ | `1.0` | `1.0` | `0.8890624999999996` | `73.48328381526854` |
| `docs5000_q96_c32768` | `5103` | `371673` | `96` | `3072` | `32768` | native TAC+PQ | `1.0` | `0.9947916666666666` | `0.8791666666666665` | `58.04598536948425` |
| `docs10000_q128_c32768` | `10135` | `739372` | `128` | `4096` | `32768` | native HNSW+PQ | `0.9895833333333334` | `0.9893973214285714` | `0.8867187500000006` | `69.01278987944805` |

Current optimized bounded native streaming row:

| Slice | Docs | Doc vectors | Queries | Query vectors | Centroids | Engine | Policy | MRR@10 | Final recall@10 vs exact | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `docs10000_q128_c32768` | `10135` | `739372` | `128` | `4096` | `32768` | native HNSW+PQ | `kc120_ef64_alpha0.35` | `0.9895833333333334` | `0.8867187500000006` | `94.92727410156428` |

Current bounded centroid-scale ladder:

| Slice | Docs | Doc vectors | Queries | Query vectors | Centroids | Engine | MRR@10 | Final recall@10 vs exact | QPS | Build note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `docs10000_q128_centroid_ladder` | `10135` | `739372` | `128` | `4096` | `65536` | native HNSW+PQ | `0.9854910714285714` | `0.8414062500000004` | `69.72040770584155` | index `92.356s`, graph `76.086s` |
| `docs10000_q128_centroid_ladder` | `10135` | `739372` | `128` | `4096` | `131072` | native HNSW+PQ | `0.9817708333333334` | `0.828125` | `77.63396980050325` | index `188.830s`, graph `142.481s` |
| `docs10000_q128_centroid_ladder` | `10135` | `739372` | `128` | `4096` | `262144` | native HNSW+PQ | `0.9856770833333334` | `0.8109374999999999` | `89.7766500938627` | index `414.644s`, graph `291.103s` |

Current larger-centroid query-policy co-sweep:

Quality floors:
- primary metric at least `0.989`
- final recall@10 vs exact at least `0.8867`, matching the fixed-policy
  `32768` baseline floor on this bounded slice

| Centroids | Row kind | `k_c` | HNSW `ef_search` | Alpha | MRR@10 | Final recall@10 vs exact | QPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `131072` | fastest | `120` | `64` | `0.3` | `0.9817708333333334` | `0.8023437499999998` | `87.69168200632623` |
| `131072` | fastest eligible after heap frontier, sort skip, visited-table sizing, and pruning-aware top-k | `200` | `64` | `0.45` | `0.9895833333333334` | `0.8906250000000002` | `69.5078171936609` |
| `262144` | fastest | `120` | `64` | `0.3` | `0.9856770833333334` | `0.7945312499999997` | `97.06290014898175` |
| `262144` | fastest eligible after heap frontier, sort skip, visited-table sizing, and pruning-aware top-k | `360` | `64` | `0.35` | `0.9934895833333334` | `0.8875000000000008` | `74.9128485720521` |

Interpretation:
- these rows are meaningful systems evidence for Kayak's local path
- they are not paper-scale reproduction rows
- judged MRR can match exact while exact top-10 overlap remains approximate,
  so "quality matched" must name the metric
- larger centroid artifacts can be fast or quality-preserving in the measured
  grid, but did not produce a quality-preserving speedup over the `32768`
  baseline
- heap-backed HNSW frontiers, the `ef == k_c` sort skip, visited-table sizing,
  and pruning-aware top-k substantially narrow the gap, but the optimized
  `32768` row is still the fastest quality-preserving bounded row

Current native HNSW+PQ internal profile:

| Slice | Full search batch s | HNSW traversal share | Rerank scoring share | Dominant isolated stage |
| --- | ---: | ---: | ---: | --- |
| `docs1500_q48_c32768` | `0.3731530674041517` | `0.6839060718283632` | `0.19329688245060372` | HNSW traversal |
| `docs10000_q128_c32768` | `1.3555743299023153` | `0.21438380846531743` | `0.6256436175858051` | rerank scoring |
| `docs10000_q128_c262144` | `1.58750085175558` | `0.4928831055891983` | `0.3770497727948497` | HNSW traversal |

Interpretation:
- candidate pruning and final top-k are not material bottlenecks on these
  slices
- optimizing batching alone is not expected to move the result meaningfully
- the next code-level work should target centroid graph traversal and sparse
  residual-PQ document scoring, with before/after profiles on the same artifacts

## What Is Not Reproduced

The following statements are not justified today:

- full MS MARCO-v1 paper result
- full LoTTE-pooled paper result
- paper-scale centroid count
- paper-scale graph build
- paper throughput comparison against Warp, IGP, or EMVB
- paper hardware/runtime parity
- Rust/kANNolo implementation parity
- SOTA
- physical or asymptotic optimality

## Blockers

1. **Corpus scale**
   - Local bounded runs are at most `10135` documents and `739372` document
     vectors.
   - Paper MS MARCO-v1 is `8.8M` passages and `598M` token vectors.

2. **Centroid scale**
   - Local bounded gates now reach `262144` centroids.
   - Paper retrieval uses about `4M` centroids for MS MARCO-v1 and about `2M`
     for LoTTE.

3. **Graph construction**
   - The current HNSW graph builder is Python reference code.
   - Paper-scale credibility requires a native or streaming graph builder with
     explicit memory and wall-clock evidence.

4. **Residual-PQ layout equivalence**
   - Kayak has residual-PQ refine paths.
   - The paper's cache-optimized Rust layout and speedups are not proven
     equivalent by Kayak's current Mojo/Python layout.

5. **External baseline parity**
   - Current local comparisons are primarily against Kayak exact and internal
     Python/native variants.
   - Paper throughput claims compare against Warp, IGP, and EMVB at metric
     cutoffs.

6. **Dataset coverage**
   - MS MARCO selected-positive bounded gates exist.
   - LoTTE-pooled paper-dataset gates do not.

7. **Metric semantics**
   - Bounded gates can match exact MRR@10 while exact top-10 overlap remains
     around `0.88-0.91`.
   - Claims must state which metric is matched.

## Claim Policy

Allowed language:

- "paper-shaped Tachiom implementation"
- "bounded-scale reproduction gate"
- "matches exact MRR@10 on selected-positive MS MARCO slices"
- "native streaming HNSW+PQ is faster than exact MaxSim on bounded local
  slices"
- "does not reproduce full paper-scale throughput"
- "USL batching was measured and is not the current material bottleneck"
- "larger-centroid query-policy rows can recover bounded exact-overlap floors,
  but not as a speedup over the `32768` baseline in the measured grid"

Disallowed language unless future evidence changes this document:

- "we reproduced the paper"
- "full Tachiom reproduction"
- "paper-equivalent throughput"
- "SOTA"
- "physically optimized"
- "full MS MARCO result"
- "LoTTE reproduced"

## Upgrade Gates

The status can be upgraded only by adding evidence for one of these gates.

### Gate 1: Larger Bounded MS MARCO

Minimum next gate:
- `50000-100000` selected-positive MS MARCO documents
- explicit document count, document-vector count, query count, query-vector
  count, and centroid count
- exact MRR@10, candidate recall@10 vs exact, final recall@10 vs exact, QPS,
  index bytes, build time

Reason:
- this tests whether current bounded behavior bends before paper scale

### Gate 2: Larger Centroid Counts

Minimum next gate:
- `262144` centroids, then `1048576` centroids if memory permits
- same judged slice and same metrics as Gate 1
- graph build time and graph bytes reported separately

Reason:
- the current `32768`-centroid evidence cannot validate the paper's `2M-4M`
  centroid regime

### Gate 3: Native Or Streaming Graph Builder

Minimum next gate:
- graph construction without Python reference bottlenecks
- bounded-memory construction report
- correctness check against the existing Python graph on a small fixture
- build-time curve over increasing centroid counts

Reason:
- Python graph construction is the largest paper-scale credibility gap

### Gate 4: Native HNSW+PQ Internal Profile

Status: implemented for the list-backed native Mojo reader.

Evidence:
- stage timings for HNSW traversal, candidate dedup/window construction,
  residual-PQ lookup/scoring, and final top-k
- measured on `docs1500_q48_c32768` and `docs10000_q128_c32768`
- profile artifacts are written as `hnsw_pq_mojo_internal_profile.json` under
  each measured streaming index root

Reason:
- USL batching did not identify a material bottleneck, so the next optimization
  needs internal timing evidence

Next use:
- measure the same artifact before and after each HNSW traversal or residual-PQ
  scoring optimization

### Gate 5: LoTTE-Pooled Slice

Minimum next gate:
- token-id-bearing LoTTE-pooled artifact
- Success@5 reported
- document/query vector counts explicit
- exact-reference overlap reported

Reason:
- the paper reports both in-domain and out-of-domain evaluation; Kayak has not
  reproduced the LoTTE side

### Gate 6: External Baseline Matrix

Minimum next gate:
- compare against at least one paper baseline family or a clearly labeled local
  proxy
- record why the comparison is or is not equivalent to Warp, IGP, and EMVB

Reason:
- Kayak exact is not a substitute for the paper's baseline comparison

## Definition Of Done For This Status

This document is current if it answers:

- what is implemented
- what has been measured
- what scale was measured
- what the paper target was
- what is still missing
- what language is allowed
- what evidence would upgrade the claim

If new benchmark rows change the boundary, update this document in the same PR
as the measurement artifact or trace note.
