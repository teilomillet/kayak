# Tachiom TAC Probe

Date: `2026-05-01`

Paper:
- **Efficient Multivector Retrieval with Token-Aware Clustering and
  Hierarchical Indexing**
- <https://arxiv.org/pdf/2604.28142>

## Claim Under Test

Tachiom's first relevant claim for Kayak is that token-aware centroid
allocation can improve multivector candidate generation by avoiding the
frequency bias of global k-means.

Implemented in this gate:
- aligned document token ids in the Python late-interaction surface
- token-aware centroid allocation and deterministic per-token clustering
- exact centroid scan for stage-1 candidate generation
- exact MaxSim rerank over the selected candidate window
- profiling counters for TAC candidate generation versus exact rerank
- paper-style Candidates Pruning using the first-stage final-rank cutoff score
- a Python and native dim128 HNSW-style centroid search reference
- an experimental TAC + i8 compressed rerank payload
- a normalized residual-PQ refine reference plus a dim128 Mojo rerank primitive
- packed/ragged `LateIndex` integration when token ids are present
- a benchmarkable Mojo dim128 TAC primitive using Python-built centroids

Not implemented in this gate:
- paper-scale MS MARCO or LoTTE throughput
- real ColBERT quality on an external judged task
- a paper-matching HNSW performance frontier

Reason:
- Kayak did not previously carry aligned document token ids. A faithful TAC
  path needed that representation contract first.
- Exact centroid scan isolates the value of token-aware allocation before graph
  approximation or residual compression effects are introduced.
- The Mojo primitive is enough to test whether TAC belongs on Kayak's native
  candidate-generation frontier before adding HNSW/PQ complexity.
- Profiling showed exact candidate rerank, not centroid scan, dominates the
  full-recall TAC points tested here. That makes compressed rerank the sound
  next implementation gate before HNSW.
- The HNSW layer was then added and measured anyway to close the paper-shape
  gap. On the current synthetic shapes it does not beat exact centroid scan.

## Implementation

Primary files:
- `python/kayak_bridge/tachiom_probe.py`
- `python/kayak_bridge/tachiom_index.py`
- `python/kayak_bridge/tachiom_allocation.py`
- `python/kayak_bridge/tachiom_clustering.py`
- `python/kayak_bridge/tachiom_metrics.py`
- `python/kayak_bridge/tachiom_mojo.py`
- `kayak/search/tachiom_tac_dim128.mojo`
- `kayak/search/tachiom_tac_profile_dim128.mojo`
- `kayak/search/tachiom_tac_profile_types_dim128.mojo`
- `python/kayak_bridge/tachiom_hnsw.py`
- `kayak/search/tachiom_tac_hnsw_dim128.mojo`
- `kayak/search/tachiom_tac_i8_dim128.mojo`
- `kayak/search/tachiom_tac_pq_dim128.mojo`
- `python/kayak_bridge/_mojo_exact_cpu_bindings.mojo`
- `python/scripts/bench_tachiom_tac_probe.py`
- `python/scripts/bench_tachiom_tac_sweep.py`
- `python/scripts/bench_tachiom_tac_pq_sweep.py`
- `python/scripts/profile_tachiom_tac_candidate_generation.py`
- `python/tests/test_tachiom_probe.py`

Representation support:
- `python/kayak_bridge/late_documents.py`
- `python/kayak_bridge/late_index.py`
- `python/kayak_bridge/late_ops.py`
- `python/kayak/encoders/colbert.py`

The ColBERT token-id path was added after inspecting the installed ColBERT
implementation locally. `Checkpoint.docFromText()` tokenizes documents and then
calls `doc(input_ids, attention_mask)`; `doc()` preserves row positions while
zeroing skipped/pad rows. Kayak now preserves document token ids when a
checkpoint exposes `doc_tokenizer` and falls back to vector-only behavior for
injected test checkpoints.

## Metrics

Candidate recall is exact top-k documents contained anywhere in the candidate
window. Final recall is ordered top-k overlap after rerank. This matters because
a candidate window is a set inspected by the reranker, not an ordered final
result.

Byte accounting now distinguishes:
- `tac_sidecar_index_bytes`: TAC centroids, token ids, offsets, and postings
- `exact_rerank_vector_bytes`: full token vectors required by exact rerank
- `index_bytes`: total bytes for the functional TAC path, equal to sidecar plus
  exact rerank vectors
- for TAC+i8, `index_bytes` counts document offsets, int8 token codes, token
  scales, centroids, centroid offsets, and centroid posting indices

Earlier sidecar-only TAC byte claims were optimistic. The tables below use
total TAC bytes for comparisons.

The TAC+i8 lane is not a PQ residual implementation from the paper. It is a
lower-risk compressed-rerank probe built on Kayak's existing int8 token scoring
machinery. It tests the measured question first: whether replacing exact
rerank vectors with a compressed payload improves the current TAC frontier.

The TAC+PQ lane stores each token as:
- assigned TAC centroid id
- residual norm
- one uint8 PQ code per residual subspace

Scoring reconstructs the token approximately as centroid contribution plus
normalized residual-PQ contribution. This matches the paper's refine-stage
shape more closely than i8, but the current implementation is still a first
native reference, not an optimized production layout.

## Validation

Focused command:

```bash
PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_encoder_api.py \
  python/tests/test_late_interaction.py \
  python/tests/test_tachiom_probe.py \
  python/tests/test_public_typing_surface.py \
  python/tests/test_public_api_contract.py \
  python/tests/test_repo_semantic_inventory.py
```

Result: `55` tests passed.

The TAC-specific tests cover:
- token-id/vector-count alignment rejection
- tail-token centroid allocation
- ragged packed `LateIndex` input
- candidate-window recall as set membership
- Mojo TAC candidate and final results matching the Python TAC reference on a
  dim128 fixture
- Mojo TAC candidate-generation profiling fields
- Python and Mojo TAC candidate pruning agreement
- Python HNSW centroid search matching exact TAC on a dense small graph
- Mojo HNSW centroid search matching the Python HNSW reference on a dim128
  fixture
- Mojo TAC+i8 candidate generation matching TAC and i8 rerank preserving the
  exact fixture result
- Python and Mojo residual-PQ rerank agreement on a dim128 fixture
- total TAC byte accounting as sidecar plus exact rerank vectors
- compressed TAC+i8 byte accounting lower than exact-rerank TAC on the fixture
- compressed residual-PQ byte accounting lower than exact-rerank TAC on the
  fixture
- deterministic residual-PQ training sampling using a smaller effective
  training set while assigning codes for all document vectors

## Smoke Result

Command:

```bash
pixi run bench_tachiom_tac_probe_mojo_smoke
```

Quiet wrapper log: `.cache/kayak/bench_quiet/20260501T122145Z`

Shape:
- documents: `96`
- document vectors/document: `48`
- total document vectors: `4608`
- queries: `4`
- query vectors/query: `12`
- total query vectors: `48`
- vector dim: `128`
- top_k: `10`
- observed token types: `250`
- top-100 token vector fraction: `0.892578125`

Result:

| System | Candidate recall@10 | Final recall@10 | Batch mean s | QPS | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy exact MaxSim | `1.0` | `1.0` | `0.0016152440002770163` | `2476.406041015472` | `2359296` |
| Python TAC exact-centroid | `1.0` | `1.0` | `0.003705533999891486` | `1079.4665492523175` | `2827280` |
| Mojo TAC exact-centroid | `1.0` | `1.0` | `0.000878894999914337` | `4551.169366522585` | `2827280` |
| Kayak i8 PLAID probe | `0.85` | `0.85` | `0.0010713434999161109` | `3733.6297838304995` | `640024` |

Mojo TAC preserved exact top-10 and was `1.84x` exact QPS on this run. Its
functional memory footprint was `1.20x` exact-vector bytes because exact rerank
vectors are still retained.

## Moderate Frontier Sweep

Command:

```bash
pixi run bench_tachiom_tac_sweep_mojo_moderate
```

Quiet wrapper log: `.cache/kayak/bench_quiet/20260501T122202Z`

Shape:
- documents: `512`
- document vectors/document: `96`
- total document vectors: `49152`
- queries: `8`
- query vectors/query: `24`
- total query vectors: `192`
- vector dim: `128`
- top_k: `10`

Best full-recall row:

| Config | Candidate recall@10 | Final recall@10 | Batch mean s | QPS | Index bytes | Build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tac_2k_cq12_k96` Mojo | `1.0` | `1.0` | `0.011021849999906408` | `725.830963047758` | `26953640` | `0.6646827150007084` |

Reference and comparison points:

| Config | Candidate recall@10 | Final recall@10 | QPS | Index bytes |
| --- | ---: | ---: | ---: | ---: |
| exact NumPy | `1.0` | `1.0` | `168.07301587568494` | `25165824` |
| `i8_256_cq48_k128` | `0.5625` | `0.5625` | `569.5808511456446` | `6800744` |
| `i8_512_cq64_k256` | `0.8125000000000001` | `0.8125000000000001` | `325.60738241495477` | `6841208` |
| `i8_1024_cq96_k512` | `1.0` | `0.9875` | `214.60489307741037` | `6869256` |
| `i8_2048_cq128_k1024` | `1.0` | `0.9875` | `214.57620160938728` | `6900296` |

Interpretation:
- Mojo TAC was the best full-recall point by QPS on this synthetic shape.
- The best full-recall TAC point was `4.32x` exact QPS and used `1.07x`
  exact-vector bytes.
- The i8 lane is still the better low-memory lane, but it did not hit exact
  final recall in this sweep.

## Large Frontier Sweep

Command:

```bash
pixi run bench_tachiom_tac_sweep_mojo_large
```

Quiet wrapper log: `.cache/kayak/bench_quiet/20260501T122222Z`

Shape:
- documents: `2048`
- document vectors/document: `128`
- total document vectors: `262144`
- queries: `8`
- query vectors/query: `32`
- total query vectors: `256`
- vector dim: `128`
- top_k: `10`

Best full-recall row:

| Config | Candidate recall@10 | Final recall@10 | Batch mean s | QPS | Index bytes | Build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tac_8k_cq32_k512` Mojo | `1.0` | `1.0` | `0.08683611000014935` | `92.1275722736341` | `142529056` | `3.787187651999375` |

Reference and comparison points:

| Config | Candidate recall@10 | Final recall@10 | QPS | Index bytes |
| --- | ---: | ---: | ---: | ---: |
| exact NumPy | `1.0` | `1.0` | `34.51159261006984` | `134217728` |
| `tac_2k_cq12_k96` Mojo | `0.38749999999999996` | `0.38749999999999996` | `435.085122910165` | `138483552` |
| `tac_4k_cq24_k256` Mojo | `0.8750000000000001` | `0.8750000000000001` | `181.38608512358653` | `139624096` |
| `i8_1024_cq96_k512` | `0.3625` | `0.3625` | `94.17595693378297` | `36504240` |
| `i8_2048_cq128_k1024` | `0.6875` | `0.6875` | `50.92991677696832` | `36611496` |

Interpretation:
- Full-recall Mojo TAC was `2.67x` exact QPS with `1.06x` exact-vector bytes.
- Smaller TAC budgets were much faster but lost recall, so candidate budget is
  still a first-class axis.
- The i8 lane remained much smaller in bytes, but did not approach full recall
  on this hard token-structured shape.

## Profiling Gate

Commands:

```bash
pixi run profile_tachiom_tac_moderate
pixi run profile_tachiom_tac_large
```

Quiet wrapper logs:
- `.cache/kayak/bench_quiet/20260501T123230Z`
- `.cache/kayak/bench_quiet/20260501T123253Z`

Measured full-recall TAC rows:
- moderate: `tac_2k_cq12_k96`
- large: `tac_8k_cq32_k512`

| Shape | Full search batch mean s | Full candidate s | Exact rerank s | Centroid selection s | Centroid scoring s | Dominant isolated stage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| moderate | `0.010037617948264277` | `0.00448565422239772` | `0.0060317318913303076` | `0.0027534488233333333` | `0.0014590920927837223` | exact rerank |
| large | `0.08564854536363636` | `0.022632930976190478` | `0.0591872893125` | `0.013371206414785824` | `0.00769925668451613` | exact rerank |

Interpretation:
- Exact rerank was about `60%` of the moderate full-search time and about
  `69%` of the large full-search time.
- HNSW over centroids would attack centroid scoring/selection, which was not
  the dominant measured stage at the full-recall operating points.
- The next implementation gate should therefore target rerank payload cost
  before introducing centroid graph complexity.

## Compressed Rerank Probe

Commands:

```bash
pixi run bench_tachiom_tac_probe_mojo_smoke
pixi run bench_tachiom_tac_sweep_i8_moderate
pixi run bench_tachiom_tac_sweep_i8_large
```

Quiet wrapper logs:
- `.cache/kayak/bench_quiet/20260501T123839Z`
- `.cache/kayak/bench_quiet/20260501T123850Z`
- `.cache/kayak/bench_quiet/20260501T123914Z`

Smoke result:

| System | Candidate recall@10 | Final recall@10 | QPS | Index bytes | Bytes/exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact NumPy | `1.0` | `1.0` | `2470.1322418230566` | `2359296` | `1.0` |
| Mojo TAC exact-rerank | `1.0` | `1.0` | `4660.01410643191` | `2827280` | `1.1983891063268228` |
| Mojo TAC+i8 | `1.0` | `0.975` | `4136.680043755548` | `1039384` | `0.4405203925238715` |
| Kayak i8 PLAID probe | `0.85` | `0.85` | `3725.2515254602854` | `640024` | `0.2712775336371528` |

Moderate result:

| Config | Candidate recall@10 | Final recall@10 | QPS | Index bytes | Bytes/exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact NumPy | `1.0` | `1.0` | `168.81151544324374` | `25165824` | `1.0` |
| `tac_2k_cq12_k96` Mojo exact-rerank | `1.0` | `1.0` | `763.694680419756` | `26953640` | `1.0710728963216145` |
| `tac_2k_cq12_k96` Mojo TAC+i8 | `1.0` | `0.9875` | `650.6529851798196` | `7882672` | `0.313223521050876` |
| `i8_1024_cq96_k512` PLAID | `1.0` | `0.9875` | `214.48020761891638` | `6869256` | `0.2729593912760417` |
| `i8_2048_cq128_k1024` PLAID | `1.0` | `0.9875` | `216.60539757210498` | `6900296` | `0.2741934458414714` |

Large result:

| Config | Candidate recall@10 | Final recall@10 | QPS | Index bytes | Bytes/exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact NumPy | `1.0` | `1.0` | `34.43067050726151` | `134217728` | `1.0` |
| `tac_8k_cq32_k512` Mojo exact-rerank | `1.0` | `1.0` | `94.28660866186556` | `142529056` | `1.0619239211082458` |
| `tac_8k_cq32_k512` Mojo TAC+i8 | `1.0` | `1.0` | `82.29498239397063` | `40817192` | `0.30411189794540405` |
| `i8_2048_cq128_k1024` PLAID | `0.6875` | `0.6875` | `50.53017108183227` | `36611496` | `0.2727658450603485` |

Interpretation:
- On the large token-structured shape, TAC+i8 reached exact final recall at
  `0.304x` exact-vector bytes and `2.39x` exact QPS.
- On the moderate shape, TAC+i8 matched the strongest PLAID final recall
  (`0.9875`) while running about `3.0x` faster, but it did not preserve exact
  final recall.
- The compressed lane is promising, but the paper-faithful compression target
  remains PQ/residual rerank rather than this int8 approximation.

## Residual-PQ Refine Probe

Command:

```bash
pixi run bench_tachiom_tac_pq_sweep_smoke
pixi run bench_tachiom_tac_probe_pq64_mojo_moderate_sample
pixi run bench_tachiom_tac_probe_pq128_mojo_moderate_sample
```

Quiet wrapper log:
- `.cache/kayak/bench_quiet/20260501T132746Z`
- `.cache/kayak/bench_quiet/20260501T132810Z`
- `.cache/kayak/bench_quiet/20260501T133021Z`

Smoke shape:
- documents: `96`
- document vectors/document: `48`
- total document vectors: `4608`
- queries: `4`
- query vectors/query: `12`
- total query vectors: `48`
- vector dim: `128`
- top_k: `10`
- TAC centroids: `768`
- centroids/query vector: `12`
- candidate_k: `48`
- PQ subspaces: `16, 32, 64, 128`
- PQ codebook size: `256`

Smoke result:

| System | Candidate recall@10 | Final recall@10 | QPS | Index bytes | Bytes/exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact NumPy | `1.0` | `1.0` | `2443.6815405891884` | `2359296` | `1.0` |
| Mojo TAC+PQ, `16` subspaces | `1.0` | `0.9249999999999999` | `1332.1843242183456` | `685072` | `0.2903713650173611` |
| Mojo TAC+PQ, `32` subspaces | `1.0` | `0.9249999999999999` | `976.5714407101473` | `758800` | `0.3216213650173611` |
| Mojo TAC+PQ, `64` subspaces | `1.0` | `0.975` | `854.7470622722658` | `906256` | `0.3841213650173611` |
| Mojo TAC+PQ, `128` subspaces | `1.0` | `1.0` | `441.7879001699325` | `1201168` | `0.5091213650173612` |

Moderate sampled-PQ result:

| System | Candidate recall@10 | Final recall@10 | QPS | Index bytes | Bytes/exact | Build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact NumPy | `1.0` | `1.0` | `166.55666986861868` | `25165824` | `1.0` | `0.0` |
| Mojo TAC+PQ, `64` subspaces, `8192` training tokens | `1.0` | `0.9500000000000001` | `142.66954387129084` | `5244840` | `0.20841121673583984` | `7.203866261001167` |
| exact NumPy for PQ128 run | `1.0` | `1.0` | `169.33458777946748` | `25165824` | `1.0` | `0.0` |
| Mojo TAC+PQ, `128` subspaces, `8192` training tokens | `1.0` | `0.9875` | `69.00519820464686` | `8390568` | `0.33341121673583984` | `9.970357170999705` |

Interpretation:
- Residual PQ now exists as both a Python reference and native dim128 rerank
  primitive, and both preserve the same candidate and final results.
- On the smoke shape, `128` residual-PQ subspaces recover exact final recall at
  `0.509x` exact-vector bytes, while `64` subspaces match the earlier TAC+i8
  smoke final recall (`0.975`) with fewer bytes (`0.384x` versus `0.441x`) but
  much lower QPS.
- On the moderate shape, deterministic `8192`-token PQ training sampling cuts
  `64`-subspace build time from an exploratory `33.6s` full-data PQ build to
  `6.6s` PQ build time, but final recall is still only `0.95` and query QPS
  remains below exact NumPy and far below TAC+i8.
- Increasing the moderate run to `128` PQ subspaces raises final recall to
  `0.9875` at `0.333x` exact-vector bytes, but QPS falls to `69.0`, so the
  current PQ path buys memory and paper shape rather than a better latency
  frontier.
- The immediate value is paper-faithfulness and a measurable compressed refine
  layout. It is not yet a better search frontier point than exact TAC or i8 on
  the measured shapes.

## Candidates Pruning And HNSW

Commands:

```bash
pixi run bench_tachiom_tac_probe_cp_mojo_smoke
pixi run bench_tachiom_tac_probe_cp_pq128_mojo_moderate_sample
pixi run bench_tachiom_tac_probe_hnsw_mojo_smoke
pixi run bench_tachiom_tac_probe_hnsw_mojo_large
```

Additional exploratory HNSW commands varied `M`, `ef_construction`,
`ef_search`, and level probability directly through
`python/scripts/bench_tachiom_tac_probe.py`.

Quiet wrapper logs:
- `.cache/kayak/bench_quiet/20260501T134925Z`
- `.cache/kayak/bench_quiet/20260501T134950Z`
- `.cache/kayak/bench_quiet/20260501T140217Z`
- `.cache/kayak/bench_quiet/20260501T140804Z`

Candidates Pruning results:

| Shape/config | Candidate recall@10 | Final recall@10 | Candidate window mean | QPS | Bytes/exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke TAC exact, `alpha=0.4` | `1.0` | `1.0` | `34.25` / `48` | `5275.0301797129205` | `1.1983574761284723` |
| smoke PQ64, `alpha=0.4` | `1.0` | `0.975` | `34.25` / `48` | `983.1450837416712` | `0.3841213650173611` |
| moderate PQ128, `alpha=0.4` | `1.0` | `0.9875` | `77.5` / `96` | `84.47930478239725` | `0.33341121673583984` |

HNSW centroid-search results:

| Shape/config | Candidate recall@10 | Final recall@10 | QPS | Bytes/exact | Build s |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke Mojo TAC exact-centroid | `1.0` | `1.0` | `4683.410247509736` | `1.1983574761284723` | `0.12107129499781877` |
| smoke Mojo HNSW, `M=16`, `ef=64`, `p=0.5` | `0.825` | `0.825` | `3187.6916839224928` | `1.2842068142361112` | `0.6361982729995361` |
| smoke Mojo HNSW, `M=32`, `ef=128`, `p=0.5` | `0.925` | `0.925` | `1198.2767580285217` | `1.3663669162326388` | `1.8951528210000106` |
| smoke Mojo HNSW, `M=48`, `ef=256`, `p=0.5` | `1.0` | `1.0` | `192.13018434463126` | `1.4468858506944444` | `3.939327193000281` |
| smoke Mojo HNSW + diversified neighbors, `M=16`, `ef=64`, `p=0.0625` | `1.0` | `1.0` | `1528.247554588097` | `1.2450425889756944` | `0.9041236969987949` |
| moderate Mojo TAC exact-centroid | `1.0` | `1.0` | `766.86818845514` | `1.071041425069173` | `0.6964201760001743` |
| moderate Mojo HNSW, `M=32`, `ef=128`, `p=0.5` | `0.775` | `0.775` | `285.0240795480723` | `1.1135670344034831` | `6.055100489998949` |
| moderate Mojo HNSW + diversified neighbors, `M=16`, `ef=64`, `p=0.0625` | `1.0` | `1.0` | `466.791558661462` | `1.0827341079711914` | `3.5379099800011318` |
| large Mojo TAC exact-centroid | `1.0` | `1.0` | `95.26289534508831` | `1.0619242191314697` | `3.7844306719998713` |
| large Mojo HNSW + diversified neighbors, `M=16`, `ef=64`, `p=0.0625` | `1.0` | `1.0` | `97.46040123047308` | `1.070725917816162` | `17.00142630099981` |

Interpretation:
- Candidates Pruning is now implemented in Python and native TAC/PQ paths. It
  reduces rerank window size without changing recall on the tested CP points.
- HNSW over TAC centroids is now implemented as a Python graph builder and
  native dim128 query-time traversal. Adding diversified neighbor pruning was
  necessary to recover recall at practical `M=16`, `ef=64` settings.
- HNSW now beats exact centroid scan slightly on the large synthetic shape
  (`97.46` versus `95.26` QPS), but build time is still much worse because
  graph construction is Python, and this remains far from the paper's reported
  scale and throughput.

## Judged Task Reproduction Gate

Implemented:
- `python/kayak_bridge/tachiom_full.py` composes TAC centroid HNSW candidate
  generation with residual-PQ rerank, giving an in-tree Python reference for
  the paper-shaped TAC + HNSW + PQ path.
- `python/kayak_bridge/tachiom_task_benchmark.py` and
  `python/scripts/bench_tachiom_task.py` run Tachiom-style retrieval on judged
  task JSON, compare against exact Kayak rankings, and report judged metrics,
  exact-overlap recall, vector counts, index bytes, and timings.
- `python/kayak_bridge/retrieval_task_builder.py`,
  `python/kayak_bridge/msmarco_passage_task.py`, and
  `python/kayak_bridge/lemb_narrativeqa_subset.py` can now write aligned
  ColBERT document token ids into task JSON when invoked with
  `include_document_token_ids` / `--include-document-token-ids`.

Validation commands:

```bash
PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_colbert_encoder.py \
  python/tests/test_tachiom_task_benchmark.py

PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_probe.py \
  python/tests/test_tachiom_task_benchmark.py
```

Results:
- `5` focused ColBERT/task-benchmark tests passed.
- `20` focused Tachiom/task-benchmark tests passed, including native Mojo
  compatibility tests and the new HNSW+PQ composition fixture.

Cached judged-artifact gate:

```bash
PYTHONPATH=python pixi run python python/scripts/bench_tachiom_task.py \
  --task .cache/kayak/lemb_narrativeqa_q8/python_task.json \
  --engine tachiom_tac \
  --tac-centroid-count 16 \
  --tac-candidate-k 10 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --output /tmp/kayak-tachiom-should-not-write.json
```

Observed result:

```text
ValueError: Tachiom task benchmark requires document token_ids aligned with document vectors
```

Interpretation:
- The cached LEMB NarrativeQA JSON has judged queries and ColBERT vectors, but
  it does not have aligned document token ids. Running TAC on it by inventing
  token ids would not reproduce the paper's token-aware clustering.
- The next faithful judged run requires rebuilding a task artifact from text
  with `--include-document-token-ids`, or building the paper datasets
  (MS MARCO, LoTTE, Wikipedia) with aligned ColBERT document token ids.
- This closes a reproduction-claim loophole: the code now refuses the
  unfaithful vector-only path instead of reporting a misleading number.

Token-id rebuild command:

```bash
PYTHONPATH=python pixi run python python/scripts/build_lemb_narrativeqa_task_json.py \
  --query-limit 8 \
  --include-document-token-ids \
  --output .cache/kayak/lemb_narrativeqa_q8_token_ids/python_task.json
```

Observed rebuilt shape:
- documents: `355`
- document vectors total: `63900`
- mean document vectors: `180`
- queries: `8`
- mean query vectors: `32`
- vector dim: `128`
- first document vector/token-id counts: `180` / `180`

Judged LEMB q8 commands:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 180 --force -- \
  env PYTHONPATH=python pixi run python python/scripts/bench_tachiom_task.py \
  --task .cache/kayak/lemb_narrativeqa_q8_token_ids/python_task.json \
  --engine tachiom_tac_hnsw \
  --exact-backend numpy_reference \
  --tac-centroid-count 2048 \
  --tac-centroids-per-query-vector 16 \
  --tac-candidate-k 128 \
  --tac-candidate-pruning-alpha 0.4 \
  --hnsw-max-neighbors 16 \
  --hnsw-ef-construction 64 \
  --hnsw-ef-search 64 \
  --warmup-iterations 1 \
  --measurement-iterations 2 \
  --output .cache/kayak/tachiom_task_lemb_q8_token_ids_hnsw/summary.json \
  --emit-quiet-mean

bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 240 --force -- \
  env PYTHONPATH=python pixi run python python/scripts/bench_tachiom_task.py \
  --task .cache/kayak/lemb_narrativeqa_q8_token_ids/python_task.json \
  --engine tachiom_tac_hnsw_pq \
  --exact-backend numpy_reference \
  --tac-centroid-count 2048 \
  --tac-centroids-per-query-vector 16 \
  --tac-candidate-k 128 \
  --tac-candidate-pruning-alpha 0.4 \
  --hnsw-max-neighbors 16 \
  --hnsw-ef-construction 64 \
  --hnsw-ef-search 64 \
  --pq-subspace-count 64 \
  --pq-codebook-size 256 \
  --pq-kmeans-iterations 6 \
  --pq-training-sample-count 8192 \
  --warmup-iterations 1 \
  --measurement-iterations 2 \
  --output .cache/kayak/tachiom_task_lemb_q8_token_ids_hnsw_pq64/summary.json \
  --emit-quiet-mean

bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 120 --force -- \
  env PYTHONPATH=python pixi run python python/scripts/bench_kayak_task_exact.py \
  --task .cache/kayak/lemb_narrativeqa_q8_token_ids/python_task.json \
  --output .cache/kayak/lemb_narrativeqa_q8_token_ids/kayak_exact_summary.json \
  --warmup-iterations 1 \
  --measurement-iterations 2
```

Judged LEMB q8 result:

| System | nDCG@10 | Exact top-10 overlap | QPS | Mean candidate window | Index bytes | Build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kayak exact packed | `0.375` | `1.0` | `697.2532492582864` | n/a | n/a | n/a |
| Tachiom TAC+HNSW+exact rerank | `0.375` | `0.8125` | `73.80957575065852` | `37.375` | `34806832` | `2.823174557999664` |
| Tachiom TAC+HNSW+PQ64 | `0.375` | `0.7999999999999999` | `31.529577443217367` | `37.375` | `6549920` | `9.963955457998964` |

Interpretation:
- This is the first judged token-id-bearing Tachiom run in this trace.
- It is not a paper reproduction: LEMB q8 is not one of the paper's reported
  MS MARCO/LoTTE/Wikipedia configurations, and the corpus is too small for
  HNSW/TAC to show the paper's systems advantage.
- The judged metric ties exact on this tiny slice, but exact-top-10 overlap is
  only about `0.80`, so the approximation is not equivalent to exact ranking.
- Kayak exact is much faster here (`697` QPS) because the corpus has only
  `355` documents and `63900` document vectors; this run is a correctness gate,
  not evidence of a production speedup.
- The PQ path cuts index bytes from `34.8M` to `6.55M`, but loses a little more
  exact-rank overlap and is slower in the current Python composition.

## MS MARCO Dev.Small Token-ID Gate

Paper target verified from `https://arxiv.org/pdf/2604.28142`:
- MS MARCO-v1: `8.8M` passages, `598M` token vectors,
  `6980` dev.small queries
- reported metric: MRR@10
- encoder: ColBERTv2
- clustering settings: `mu=128`, `tau=256`, `epsilon=4`, `theta=39`,
  `10` k-means iterations, PQ32 with 8-bit codes
- retrieval settings: MS MARCO uses about `4M` centroids, HNSW `M=32`,
  `ef_construction=1500`, and grid-searches `k_c`, `k_d`, and `alpha`

Verified local resource constraint:
- available disk before this gate: `295G`
- materializing the paper's `598M x 128 x float32` token vectors would require
  about `306G` for vectors alone, before token ids, offsets, metadata, JSON
  overhead, centroids, postings, residuals, or temporary build memory
- therefore the current JSON task path cannot honestly reproduce the full
  paper-scale artifact; it can only run bounded paper-dataset gates

Official MS MARCO files were downloaded through `ir_datasets` into
`.cache/ir_datasets`:
- `.cache/ir_datasets/msmarco-passage/collection.tsv`
- `.cache/ir_datasets/msmarco-passage/dev/small/queries.tsv`
- `.cache/ir_datasets/msmarco-passage/dev/small/qrels`

Builder fix made before running:
- MS MARCO task JSON now uses `primary_metric="mrr"` instead of `recall`
  because the paper and official dev.small evaluation target MRR@10
- document encoding is batched through the existing ColBERT batch path
- `bench_tachiom_task.py` now exposes `--tac-active-token-floor`, so paper
  `epsilon=4` can be set explicitly

Artifact command:

```bash
env IR_DATASETS_HOME=/home/teilo/Code/kayak/.cache/ir_datasets \
  PYTHONPATH=python pixi run python \
  python/scripts/build_msmarco_passage_task_json.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --document-limit 1000 \
  --query-limit 32 \
  --include-document-token-ids \
  --document-batch-size 16 \
  --dataset-id msmarco-passage/dev/small \
  --output \
  .cache/kayak/msmarco_passage_dev_small_docs1000_q32_token_ids/python_task.json
```

Observed artifact shape:
- source documents scanned: `8841823`
- selected documents: `1036` (`1000` prefix docs plus `36` required positives)
- selected queries: `32`
- missing selected positives: `0`
- document vectors total: `74804`
- mean document vectors: `72`
- query vectors total: `1024`
- mean query vectors: `32`
- vector dim: `128`
- JSON size: `289821668` bytes (`276MiB`)

Benchmark commands:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 120 --force -- \
  env PYTHONPATH=python pixi run python \
  python/scripts/bench_kayak_task_exact.py \
  --task \
  .cache/kayak/msmarco_passage_dev_small_docs1000_q32_token_ids/python_task.json \
  --output \
  .cache/kayak/msmarco_passage_dev_small_docs1000_q32_token_ids/kayak_exact_summary.json \
  --warmup-iterations 1 \
  --measurement-iterations 3

bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 180 --force -- \
  env PYTHONPATH=python pixi run python \
  python/scripts/bench_tachiom_task.py \
  --task \
  .cache/kayak/msmarco_passage_dev_small_docs1000_q32_token_ids/python_task.json \
  --engine tachiom_tac_hnsw_pq \
  --exact-backend numpy_reference \
  --tac-centroid-count 8192 \
  --tac-micro-token-threshold 128 \
  --tac-small-token-threshold 256 \
  --tac-active-token-floor 4 \
  --tac-min-vectors-per-centroid 39 \
  --tac-kmeans-iterations 10 \
  --tac-centroids-per-query-vector 40 \
  --tac-candidate-k 250 \
  --tac-candidate-pruning-alpha 0.4 \
  --hnsw-max-neighbors 32 \
  --hnsw-ef-construction 1500 \
  --hnsw-ef-search 60 \
  --pq-subspace-count 32 \
  --pq-codebook-size 256 \
  --pq-kmeans-iterations 10 \
  --pq-training-sample-count 32768 \
  --warmup-iterations 1 \
  --measurement-iterations 3 \
  --output \
  .cache/kayak/msmarco_passage_dev_small_docs1000_q32_token_ids/tachiom_hnsw_pq32_paperish_summary.json \
  --emit-quiet-mean
```

Judged MS MARCO dev.small bounded result:

| System | MRR@10 | Exact top-10 overlap | QPS | Mean candidate window | Index bytes | Build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kayak exact packed | `1.0` | `1.0` | `489.19` | n/a | n/a | n/a |
| Tachiom TAC+HNSW+PQ32 | `1.0` | `0.8625` | `21.61592820033478` | `47.78125` | `10241008` | `105.44975530999909` |

Interpretation:
- This is the first token-id-bearing MS MARCO dev.small gate, with the paper's
  TAC allocation parameters and PQ32/HNSW shape on a bounded corpus.
- It is not a full paper reproduction: the bounded task has `1036` documents
  and `74804` document vectors, versus the paper target's `8.8M` passages and
  `598M` token vectors.
- The perfect judged MRR is not strong external quality evidence because the
  bounded artifact forces selected positives into the document subset.
- The approximation is still not exact-ranking equivalent: top-10 overlap with
  exact is `0.8625` at this grid point.
- Exact remains much faster on this small corpus because packed exact search
  has little work to do, while this reference Tachiom path still pays Python
  HNSW/PQ build and query overhead.

Larger bounded MS MARCO command:

```bash
env IR_DATASETS_HOME=/home/teilo/Code/kayak/.cache/ir_datasets \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=python pixi run python \
  python/scripts/build_msmarco_passage_task_json.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --document-limit 5000 \
  --query-limit 128 \
  --include-document-token-ids \
  --document-batch-size 16 \
  --dataset-id msmarco-passage/dev/small \
  --output \
  .cache/kayak/msmarco_passage_dev_small_docs5000_q128_token_ids/python_task.json
```

Larger bounded artifact shape:
- source documents scanned: `8841823`
- selected documents: `5135` (`5000` prefix docs plus `135` required positives)
- selected queries: `128`
- missing selected positives: `0`
- document vectors total: `373889`
- mean document vectors: `73`
- query vectors total: `4096`
- mean query vectors: `32`
- vector dim: `128`
- JSON size: `1443351393` bytes

Larger bounded MS MARCO result:

| System | Centroids | MRR@10 | Exact top-10 overlap | QPS | Mean candidate window | Index bytes | Build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kayak exact packed | n/a | `0.99609375` | `1.0` | `156.72` | n/a | n/a | n/a |
| HNSW+PQ32, `k_c=40`, `k_d=250`, `alpha=0.4` | `8192` | `0.94921875` | `0.8304687500000006` | `18.89544433779873` | `107.2578125` | `25060812` | `114.7227611569997` |
| HNSW+PQ32, `k_c=120`, `k_d=1000`, `alpha=0.35` | `8192` | `0.95703125` | `0.8367187500000004` | `10.106178381254281` | `115.6484375` | `25060812` | `114.490836029001` |
| Exact-centroid+PQ32, `k_c=120`, `k_d=1000`, `alpha=0.35` | `8192` | `0.95703125` | `0.8406250000000004` | `28.85201063958941` | `116.9140625` | `22765276` | `33.61223868700108` |
| Exact-centroid+exact-rerank, `k_c=120`, `k_d=1000`, `alpha=0.35` | `8192` | `0.95703125` | `0.8835937500000002` | `43.966689775813585` | `116.9140625` | `200670904` | `3.0977908879995084` |
| Exact-centroid+exact-rerank, `k_c=120`, `k_d=1000`, `alpha=0.35` | `32768` | `0.99609375` | `0.9281250000000003` | `15.254470490852876` | `78.90625` | `213613952` | `21.122563268001613` |
| Exact-centroid+PQ32, `k_c=120`, `k_d=1000`, `alpha=0.35` | `32768` | `0.99609375` | `0.88671875` | `13.18817967379964` | `78.90625` | `35511716` | `43.48234642599891` |
| HNSW+PQ32, `k_c=120`, `k_d=1000`, `alpha=0.35` | `32768` | `0.99609375` | `0.8890625000000001` | `10.348755931720849` | `80.71875` | `44702052` | `444.01621701399927` |

Larger bounded interpretation:
- On this 5k-document MS MARCO gate, `8192` centroids are not enough to match
  exact judged quality; exact rerank at the same candidate set confirms the
  loss is in centroid-only gather, not PQ refine.
- Raising TAC to `32768` centroids matches exact MRR@10 on this bounded gate
  for exact rerank, exact-centroid+PQ32, and HNSW+PQ32.
- The result is still not a full paper reproduction: the gate has `373889`
  document vectors, while the paper reports MS MARCO at `598M` token vectors
  and about `4M` centroids.
- The current in-tree reference does not reproduce the paper's speedup:
  exact packed search is still about `156.7` QPS, while the best quality-matched
  HNSW+PQ32 run is about `10.35` QPS. This is expected evidence that the Python
  HNSW/PQ composition and JSON artifact path are not the paper's Rust/kANNolo
  performance path.
- The quality result is meaningful as a gate: with sufficient centroid budget,
  the paper-shaped algorithm can match exact judged MRR on a token-id-bearing
  MS MARCO subset while using `44.7M` index bytes instead of carrying the exact
  rerank vectors in the TAC index.

## Paper-Scale Materialization Path

The JSON task path cannot materialize full MS MARCO on this machine because it
stores float vectors as text and previously kept all encoded rows in one JSON
object. A new sharded binary materializer is available at:

- `python/kayak_bridge/encoded_snapshot.py`
- `python/kayak_bridge/msmarco_colbert_snapshot.py`
- `python/scripts/materialize_msmarco_colbert_snapshot.py`

Format:
- `manifest.json` at the snapshot root
- `shards/000000/vectors.f16`: contiguous document token vectors
- `shards/000000/token_ids.u32`: aligned document token ids
- `shards/000000/doc_offsets.u64.npy`: document row offsets into the shard
- `shards/000000/doc_ids.txt`: document ids in shard order
- `shards/000000/stats.json`: per-shard vector counts and byte counts
- `queries/query_vectors.f32`, `queries/query_offsets.u64.npy`, and
  `queries/queries.jsonl` for judged query vectors and qrels metadata

Paper-scale byte estimate command:

```bash
env PYTHONPATH=python pixi run python \
  python/scripts/materialize_msmarco_colbert_snapshot.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --estimate-paper-scale
```

Observed estimate for MS MARCO paper scale with `float16` document vectors and
`uint32` token ids:
- vectors: `153088000000` bytes
- token ids: `2392000000` bytes
- document offsets: `70734592` bytes
- total document payload: `155550734592` bytes, about `145GiB`
- verified free disk before the change: about `293G`

Full materialization command:

```bash
env IR_DATASETS_HOME=/home/teilo/Code/kayak/.cache/ir_datasets \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=python pixi run python \
  python/scripts/materialize_msmarco_colbert_snapshot.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --document-batch-size 16 \
  --query-batch-size 64 \
  --shard-max-vectors 4000000 \
  --dataset-id msmarco-passage/dev/small \
  --resume
```

The `--resume` flag skips completed shard directories and writes only new
shards. This matters because CPU ColBERT encoding for `8.8M` passages is a long
job even though the disk layout now fits.

Smoke command:

```bash
env IR_DATASETS_HOME=/home/teilo/Code/kayak/.cache/ir_datasets \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=python pixi run python \
  python/scripts/materialize_msmarco_colbert_snapshot.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke \
  --document-limit 128 \
  --query-limit 8 \
  --document-batch-size 16 \
  --query-batch-size 8 \
  --shard-max-vectors 4096 \
  --dataset-id msmarco-passage/dev/small
```

Smoke result:
- `128` documents, `8752` document vectors, `8` queries
- `3` shards with `2276568` document payload bytes
- shard raw vector/token-id counts were checked against `stats.json`
- resume was tested by extending the same snapshot to `160` documents; the
  manifest then reported `4` shards and `10902` document vectors

Interpretation:
- The machine can now materialize the paper-scale document vectors on disk if
  `float16` document vectors are acceptable for the snapshot boundary.
- The full encoding run has not been completed yet; it is an hours-to-days CPU
  job, but it is now resumable and does not require an impossible full-corpus
  JSON artifact.

## Snapshot-to-Tachiom Bridge

The next bridge now exists:
- `python/kayak_bridge/encoded_snapshot_loader.py`
- `python/kayak_bridge/tachiom_snapshot_benchmark.py`
- `python/scripts/bench_tachiom_snapshot.py`

What it does:
- loads selected document rows from `vectors.f16`, `token_ids.u32`,
  `doc_offsets.u64.npy`, and `doc_ids.txt` directly into packed arrays
- loads query vectors and qrels metadata from the query sidecar
- builds TAC, TAC+HNSW, TAC+PQ, or TAC+HNSW+PQ without a task JSON artifact
- keeps a `--max-vector-count` guard because the current Tachiom reference path
  still needs one in-memory compute matrix for the selected documents
- exposes `iter_snapshot_document_shards(...)` as a memmap shard iterator for
  the next streaming TAC/PQ builder

Smoke command:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 60 --force -- \
  env PYTHONPATH=python pixi run python \
  python/scripts/bench_tachiom_snapshot.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke \
  --engine tachiom_tac_hnsw_pq \
  --document-limit 128 \
  --query-limit 4 \
  --max-vector-count 20000 \
  --tac-centroid-count 256 \
  --tac-centroids-per-query-vector 16 \
  --tac-candidate-k 64 \
  --tac-candidate-pruning-alpha 0.35 \
  --hnsw-max-neighbors 8 \
  --hnsw-ef-construction 32 \
  --hnsw-ef-search 32 \
  --pq-subspace-count 32 \
  --pq-codebook-size 64 \
  --pq-kmeans-iterations 4 \
  --pq-training-sample-count 4096 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --output \
  .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke/tachiom_snapshot_hnsw_pq_summary.json \
  --emit-quiet-mean
```

Smoke result:
- source vector dtype: `float16`
- source token-id dtype: `uint32`
- loaded documents: `128`
- loaded document vectors: `8752`
- loaded snapshot payload bytes: `2276568`
- engine: `tachiom_tac_hnsw_pq`
- index bytes: `593544`
- centroid count: `256`
- posting count: `2984`
- QPS: `158.44193273487895`

The judged MRR is `0.0` on this tiny prefix because the smoke snapshot is the
first `128` collection rows and does not force qrels positives into the
selected documents. That is expected for this bridge test; the verified claim
is that Tachiom can now build and search directly from snapshot shards without
the JSON task materialization path.

Interpretation:
- This bridge removes JSON/Python-list materialization for selected snapshot
  rows, but the selected-slice benchmark still loads those selected rows into
  one compute matrix before building the current in-memory TAC reference.
- It was therefore a necessary bridge, not the final paper-scale index builder.

## Streaming TAC/PQ Artifact Builder

The streaming builder now exists:
- `python/kayak_bridge/tachiom_streaming_index.py`
- `python/kayak_bridge/tachiom_streaming_search.py`
- `python/kayak_bridge/tachiom_streaming_hnsw.py`
- `python/kayak_bridge/tachiom_streaming_benchmark.py`
- `python/scripts/build_tachiom_streaming_index.py`
- `python/scripts/build_tachiom_streaming_hnsw.py`
- `python/scripts/bench_tachiom_streaming_index.py`
- `python/tests/test_tachiom_streaming_index.py`

What it does:
- scans snapshot shard memmaps to collect per-token counts, sums, and spread
  statistics without constructing a full document-token matrix
- allocates TAC centroids from those per-token statistics
- writes deterministic per-token centroid training samples to a memmap
- trains and writes `centroids.f32` and `centroid_token_ids.i64`
- streams the snapshot again to write centroid/document postings as CSR files
- writes residual-PQ query-time payloads in original document-token order:
  `token_centroid_positions.u32`, `residual_norms.f32`, and `pq_codes.u8`
- writes `doc_ids.txt` and `doc_offsets.u64.npy` for query-time document
  reconstruction
- loads the resulting artifact with memory-mapped query-time arrays and can
  run exact-centroid TAC candidate generation plus residual-PQ rerank
- optionally builds and persists a centroid HNSW sidecar next to the streaming
  artifact, then searches the artifact through that graph and the same
  residual-PQ rerank payload

Deliberate implementation choice:
- exact per-token Lloyd k-means over all `598M` vectors is not attempted in
  Python; large token partitions use deterministic sampled centroids when the
  bounded k-means limits would be exceeded
- this is a feasibility choice for this machine, not a paper-equivalence claim
- the persisted HNSW graph uses the existing Python reference builder, so it
  validates graph persistence and traversal but does not yet prove paper-scale
  graph construction throughput

Streaming build smoke command:

```bash
env PYTHONPATH=python pixi run python \
  python/scripts/build_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke \
  --output \
  .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke/streaming_tachiom_index \
  --overwrite \
  --tac-centroid-count 256 \
  --tac-micro-token-threshold 4 \
  --tac-small-token-threshold 16 \
  --tac-active-token-floor 1 \
  --tac-min-vectors-per-centroid 4 \
  --tac-kmeans-iterations 4 \
  --tac-centroids-per-query-vector 16 \
  --tac-candidate-k 64 \
  --tac-candidate-pruning-alpha 0.35 \
  --centroid-samples-per-centroid 2 \
  --kmeans-max-centroids-per-token 64 \
  --kmeans-max-samples-per-token 512 \
  --assignment-vector-chunk-size 512 \
  --assignment-centroid-chunk-size 512 \
  --posting-partition-count 8 \
  --pq-subspace-count 32 \
  --pq-codebook-size 64 \
  --pq-kmeans-iterations 4 \
  --pq-training-sample-count 4096
```

Streaming build smoke result:
- source snapshot: `160` documents, `10902` document vectors, `4` shards,
  `8` query sidecars
- centroid count: `256`
- token type count: `163`
- centroid training sample vectors: `511`
- posting count: `3581`
- query-time index payload bytes: `620422`
- build payload bytes including retained centroid samples: `882054`

Structural validation:
- `centroids.f32`: `131072` bytes, matching `256 x 128 x float32`
- `centroid_token_ids.i64`: `2048` bytes, matching `256` centroid token ids
- `token_centroid_positions.u32`: `43608` bytes, matching `10902` vectors
- `residual_norms.f32`: `43608` bytes, matching `10902` vectors
- `pq_codes.u8`: `348864` bytes, matching `10902 x 32` PQ codes
- `pq_codebooks.f32`: `32768` bytes, matching `32 x 64 x 4 x float32`
- posting offsets end at `3581`, matching the posting file length
- document offsets end at `10902`, matching the document vector count

Streaming query smoke command:

```bash
env PYTHONPATH=python pixi run python \
  python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke \
  --index \
  .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke/streaming_tachiom_index \
  --query-limit 4 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --run-exact \
  --max-exact-vector-count 20000 \
  --output \
  .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke/streaming_tachiom_index/search_smoke_summary.json \
  --emit-quiet-mean
```

Streaming query smoke result:
- query count: `4`
- query vectors total: `128`
- query-time index bytes: `620422`
- QPS: `171.23368302577765`
- candidate window mean: `37.25`
- candidate recall@10 versus exact on the same `160`-doc prefix:
  `0.7499999999999999`
- final recall@10 versus exact: `0.7499999999999999`
- judged MRR@10: `0.0`

Persisted HNSW sidecar command:

```bash
env PYTHONPATH=python pixi run python \
  python/scripts/build_tachiom_streaming_hnsw.py \
  --index \
  .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke/streaming_tachiom_index \
  --overwrite \
  --hnsw-max-neighbors 16 \
  --hnsw-ef-construction 64 \
  --hnsw-ef-search 64 \
  --hnsw-level-probability 0.0625
```

Persisted HNSW sidecar result:
- centroid count: `256`
- level count: `4`
- entry point: `172`
- graph edges: `4408`
- graph bytes: `28192`
- graph build seconds: `0.24236201099847676`

Streaming HNSW query smoke command:

```bash
env PYTHONPATH=python pixi run python \
  python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke \
  --index \
  .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke/streaming_tachiom_index \
  --engine streaming_tac_hnsw_pq \
  --query-limit 4 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --run-exact \
  --max-exact-vector-count 20000 \
  --output \
  .cache/kayak/msmarco_colbertv2_f16_snapshot_smoke/streaming_tachiom_index/search_hnsw_smoke_summary.json \
  --emit-quiet-mean
```

Streaming HNSW query smoke result:
- query-time index bytes including graph: `648614`
- QPS: `38.600813537908564`
- candidate window mean: `37.25`
- candidate recall@10 versus exact on the same `160`-doc prefix:
  `0.7499999999999999`
- final recall@10 versus exact: `0.7499999999999999`
- judged MRR@10: `0.0`

Interpretation:
- The judged MRR remains `0.0` for the same reason as the bridge smoke: this
  prefix snapshot does not force qrels positives into the selected documents.
- The verified claim is narrower and concrete: Kayak can now build and search
  a TAC/PQ artifact from binary MS MARCO snapshot shards without task JSON,
  Python vector lists, or a full in-RAM document-token matrix.
- The search result is not a quality reproduction. The smoke used only `256`
  centroids, not the paper's MS MARCO-scale budget of about `4M` centroids.
- The HNSW artifact is also not yet a paper-throughput reproduction: on this
  tiny centroid count, Python graph traversal is slower than exact centroid
  scan, and graph construction has not been validated at paper-scale centroid
  counts.

## Judged-Positive Streaming Snapshot Gate

The prefix-only smoke above validates mechanics, but it does not validate
judged quality because the selected documents do not necessarily include qrels
positives. The snapshot materializer now has an explicit bounded-slice mode:

- `python/kayak_bridge/msmarco_colbert_snapshot.py` accepts
  `include_query_positives`
- `python/scripts/materialize_msmarco_colbert_snapshot.py` exposes
  `--include-query-positives`
- `python/tests/test_msmarco_colbert_snapshot.py` verifies that a bounded
  prefix is extended with selected-query positive documents outside the prefix

Judged-positive snapshot command:

```bash
env IR_DATASETS_HOME=/home/teilo/Code/kayak/.cache/ir_datasets \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=python pixi run python \
  python/scripts/materialize_msmarco_colbert_snapshot.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot_judged_smoke \
  --document-limit 256 \
  --query-limit 8 \
  --include-query-positives \
  --document-batch-size 16 \
  --query-batch-size 8 \
  --shard-max-vectors 4096 \
  --dataset-id msmarco-passage/dev/small
```

Judged-positive snapshot shape:
- documents: `264` (`256` prefix documents plus `8` required positives)
- document vectors: `18846`
- shards: `5`
- queries: `8`
- query vectors: `256`
- required positive doc count: `8`
- payload bytes: `4902112`

Two streaming TAC/PQ budgets were tested on this judged snapshot:

| Engine | Centroids | MRR@10 | Exact MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact | QPS | Index bytes | Graph build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-centroid TAC+PQ32 | `1024` | `0.875` | `1.0` | `0.8` | `0.7625` | `78.09259106587905` | `1363666` | n/a |
| HNSW TAC+PQ32 | `1024` | `0.875` | `1.0` | `0.8125` | `0.7875` | `23.99632445897621` | `1474826` | `1.2846787620001123` |
| exact-centroid TAC+PQ32 | `4096` | `1.0` | `1.0` | `0.9500000000000001` | `0.8875` | `40.16663012730872` | `3003798` | n/a |
| HNSW TAC+PQ32 | `4096` | `1.0` | `1.0` | `0.925` | `0.875` | `14.681758792846209` | `3446454` | `5.910388268999668` |

Interpretation:
- This closes the first judged streaming quality gate: with `4096` centroids,
  the streaming TAC/PQ artifact matches exact MRR@10 on the bounded
  judged-positive MS MARCO slice.
- It does not prove exact ranking equivalence; the `4096`-centroid final
  top-10 overlap is `0.8875`.
- It does not prove paper throughput; on this small slice, exact centroid scan
  is still faster than the Python HNSW traversal. HNSW is expected to matter at
  much larger centroid counts, but that expectation is not yet measured here.
- The candidate gather loss seen at `1024` centroids reproduces the earlier
  bounded JSON observation: the centroid budget is a first-class quality axis.

Larger judged-positive streaming command:

```bash
env IR_DATASETS_HOME=/home/teilo/Code/kayak/.cache/ir_datasets \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=python pixi run python \
  python/scripts/materialize_msmarco_colbert_snapshot.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot_judged_docs1000_q32 \
  --document-limit 1000 \
  --query-limit 32 \
  --include-query-positives \
  --document-batch-size 16 \
  --query-batch-size 16 \
  --shard-max-vectors 8192 \
  --dataset-id msmarco-passage/dev/small
```

Larger judged-positive streaming shape:
- documents: `1036` (`1000` prefix documents plus required positives)
- document vectors: `74804`
- shards: `10`
- queries: `32`
- query vectors: `1024`
- required positive doc count: `36`
- payload bytes: `19457408`

The streaming index used `8192` TAC centroids, PQ32, 8-bit codebooks, `k_c=40`,
`k_d=250`, and `alpha=0.4`.

| Engine | Centroids | MRR@10 | Exact MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact | QPS | Index bytes | Graph build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-centroid TAC+PQ32 | `8192` | `1.0` | `1.0` | `0.8625` | `0.846875` | `31.24518300374927` | `7621550` | n/a |
| HNSW TAC+PQ32 | `8192` | `1.0` | `1.0` | `0.8468749999999999` | `0.8281249999999999` | `13.296070690802747` | `8507454` | `14.525193268000294` |
| exact-centroid TAC+PQ32 | `32768` | `1.0` | `1.0` | `0.9687499999999999` | `0.921875` | `11.457145753310499` | `20665722` | n/a |
| HNSW TAC+PQ32 | `32768` | `1.0` | `1.0` | `0.9374999999999999` | `0.9093749999999998` | `10.347699563455704` | `24208922` | `37.931196748999355` |

Interpretation:
- This reproduces the earlier `docs1000_q32` JSON judged-quality gate through
  the streaming snapshot/index path instead of task JSON.
- The bounded judged metric is closed: both streaming engines match exact
  MRR@10 on this `1036`-document, `74804`-vector slice.
- Retrying with `32768` centroids improves exact top-10 overlap from `0.846875`
  to `0.921875` for exact-centroid TAC+PQ32, but reduces QPS from `31.25` to
  `11.46` and raises index bytes from `7.62M` to `20.67M`.
- It still is not a paper-scale result. The paper MS MARCO target is `8.8M`
  passages, `598M` token vectors, `6980` queries, and about `4M` centroids.
- HNSW remains slower than exact centroid scan at `8192` centroids in this
  Python reference path and is only slightly faster than exact centroid scan at
  `32768` centroids, so the missing paper-throughput evidence is now
  specifically native/optimized graph traversal at much larger centroid counts.

The repeatable scale-gate runner now exists:

- `python/scripts/run_tachiom_streaming_scale_gate.py`

It materializes a judged-positive snapshot, builds the streaming TAC/PQ index,
builds the HNSW graph unless disabled, benchmarks both paths against exact, and
writes one `scale_gate_summary.json`.

Scale-gate command for a new size:

```bash
env IR_DATASETS_HOME=/home/teilo/Code/kayak/.cache/ir_datasets \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=python pixi run python \
  python/scripts/run_tachiom_streaming_scale_gate.py \
  --collection .cache/ir_datasets/msmarco-passage/collection.tsv \
  --queries .cache/ir_datasets/msmarco-passage/dev/small/queries.tsv \
  --qrels .cache/ir_datasets/msmarco-passage/dev/small/qrels \
  --output-root .cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768 \
  --document-limit 1500 \
  --query-limit 48 \
  --document-batch-size 16 \
  --query-batch-size 16 \
  --shard-max-vectors 8192 \
  --overwrite \
  --max-exact-vector-count 150000 \
  --tac-centroid-count 32768 \
  --tac-centroids-per-query-vector 120 \
  --tac-candidate-k 1000 \
  --tac-candidate-pruning-alpha 0.35 \
  --pq-subspace-count 32 \
  --pq-codebook-size 256 \
  --pq-training-sample-count 32768 \
  --warmup-iterations 1 \
  --measurement-iterations 3
```

New-size scale-gate result:

| Engine | Docs | Doc vectors | Queries | Centroids | MRR@10 | Exact MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact | QPS | Index bytes | Graph build s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-centroid TAC+PQ32 | `1553` | `114393` | `48` | `32768` | `1.0` | `1.0` | `0.9562499999999998` | `0.9104166666666663` | `11.370231994571297` | `22361484` | n/a |
| HNSW TAC+PQ32 | `1553` | `114393` | `48` | `32768` | `1.0` | `1.0` | `0.9458333333333333` | `0.9041666666666663` | `9.560129231202735` | `25904684` | `38.60979485899952` |

Interpretation:
- The `docs1500_q48` scale gate preserves the bounded judged-quality result:
  both streaming paths match exact MRR@10.
- Exact top-10 overlap remains below exact but stable around `0.91`.
- HNSW is still slower than exact centroid scan at this size in Python. This
  confirms that the next support requirement is native/optimized graph
  traversal, not more Python orchestration.

## Streaming Query-Time Speed Pass

Claim:
- the `docs1500_q48_c32768` exact-centroid streaming reader was bottlenecked
  by top-k centroid selection and many tiny posting updates, not by centroid
  matmul or residual-PQ rerank.

Evidence before editing, measured on the same `1553`-document, `114393`-vector,
`48`-query, `32768`-centroid artifact:
- candidate generation: `3.805230317007954s`
- residual-PQ rerank: `0.4262729099827993s`
- candidate-stage split: matmul `0.13556920899281977s`, top-k selection
  `2.908541208937095s`, posting accumulation `0.9827112530037994s`, final
  candidate ranking `0.004739359999803128s`

Change:
- `_top_positions` now uses `argpartition` for the bounded top-k case and then
  sorts only the selected/tied boundary positions to keep deterministic
  low-position tie breaks.
- streaming TAC posting accumulation now expands the selected centroid posting
  rows once per query token and applies a single `np.maximum.at`, instead of
  issuing one small NumPy maximum update per centroid posting row.

Reason:
- the optimized helper keeps the same ranking semantics while avoiding a full
  sort over all `32768` centroids for every query token.
- the posting update keeps the same per-query-token max-over-centroids score,
  but reduces Python-to-NumPy call overhead when selected postings are short.

After-edit profile on the same artifact:
- candidate generation: `0.8751495370051998s`
- residual-PQ rerank: `0.4086024629868916s`
- candidate counts unchanged: min `19`, mean `47.041666666666664`, max `85`

After exact-centroid/posting edit benchmark rerun:

| Engine | MRR@10 | Exact MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact | QPS before | QPS after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-centroid TAC+PQ32 | `1.0` | `1.0` | `0.9562499999999998` | `0.9104166666666663` | `11.370231994571297` | `37.649853131691216` |
| HNSW TAC+PQ32 | `1.0` | `1.0` | `0.9458333333333333` | `0.9041666666666663` | `9.560129231202735` | `10.615960412484354` |

Interpretation:
- the exact-centroid reader is now `3.31x` faster on this bounded artifact
  without changing judged MRR or exact top-10 overlap.
- the HNSW reader improves only `1.11x`, so its remaining bottleneck is graph
  traversal/search overhead rather than the shared posting accumulator.
- this is still not paper-throughput evidence. It is a measured local
  query-time optimization for the bounded streaming reproduction gate.

Follow-up HNSW traversal pass:
- focused HNSW candidate-stage split showed graph traversal at
  `2.7500895919874893s` of `3.003929042170057s`; selected-centroid score
  calculation was only `0.014147132143989438s`, posting accumulation
  `0.23417871202764218s`, and final candidate ranking
  `0.005513606010936201s`.
- the persisted HNSW reader now batches neighbor dot products in greedy and
  layer search while keeping the same heap policy and visited-node semantics.
- focused graph traversal time fell to `1.603570299132116s` for the same
  `1536` query vectors.
- end-to-end HNSW TAC+PQ32 improved from the post-posting baseline
  `10.615960412484354` QPS to `15.765986` QPS, with MRR@10 still `1.0`,
  candidate recall@10 vs exact still `0.9458333333333333`, and final recall@10
  vs exact still `0.9041666666666663`.

Follow-up HNSW build pass:
- the in-memory HNSW graph builder now batches neighbor score computation in
  greedy search, layer search, and neighbor pruning.
- rebuilding the same `32768`-centroid graph into a temporary sidecar reduced
  build time from `38.60979485899952s` to `35.63000441500117s`.
- the rebuilt graph kept the same `558040` edge count and `3543200` graph
  bytes. Its neighbor file was not byte-identical, so it was benchmarked
  separately: MRR@10 stayed `1.0`, candidate recall@10 vs exact stayed
  `0.9458333333333333`, final recall@10 vs exact stayed
  `0.9041666666666663`, and QPS was `15.868569`.

Native streaming TAC/PQ pass:
- `streaming_tac_pq_mojo` now prepares the existing streaming memmap artifact
  into the native dim128 residual-PQ TAC primitive instead of routing query-time
  scoring through Python loops.
- the streaming benchmark harness now groups same-shape query matrices and
  calls `search_batch_positions` once per shape group. Reason: the public API
  is already batched, and the Mojo primitive only exposes its real throughput
  when the benchmark does not wrap every query in a separate Python call.
- on the same `docs1500_q48_c32768` artifact, the new native engine produced
  exactly the same final rows as the Python streaming TAC+PQ path for all `48`
  queries.
- with `2` warmup and `5` measurement iterations, native exact-centroid
  TAC+PQ32 reached `73.056123` QPS with MRR@10 `1.0`, exact MRR@10 `1.0`,
  candidate recall@10 vs exact `0.9562499999999998`, and final recall@10 vs
  exact `0.9104166666666663`.
- exact MaxSim on the same snapshot (`1553` docs, `114393` doc vectors, `48`
  queries, `1536` query vectors) measured `47.25909877760549` QPS with the
  same warmup/measurement counts.

Interpretation:
- the native streaming TAC+PQ path is now `1.55x` faster than exact MaxSim on
  this bounded judged-positive slice while matching judged MRR@10.
- this is not a SOTA claim: final top-10 overlap is still approximate, HNSW is
  not the faster path on this slice, and no full MS MARCO/LoTTE paper-scale run
  has completed.

Native streaming HNSW/PQ sparse rerank pass:
- `streaming_tac_hnsw_pq_mojo` now prepares the same streaming TAC/PQ artifact
  plus the persisted HNSW sidecar into a native dim128 query-time engine.
- the native rerank path intentionally does not build the full
  `query_vectors x centroid_count` table. It scores centroid components only
  for the document tokens inside the candidate window, while still using the
  residual-PQ lookup table for the query. Reason: exact centroid-table rerank
  would reintroduce centroid-count scaling after HNSW candidate generation.
- on `docs1500_q48_c32768`, native HNSW+PQ sparse rerank reached
  `125.52333181000189` QPS, with MRR@10 `1.0`, candidate recall@10 vs exact
  `0.94375`, and final recall@10 vs exact `0.9041666666666663`.
- the same artifact's Python persisted-HNSW reader measured `15.891882` QPS
  with candidate recall@10 vs exact `0.9458333333333333` and the same final
  recall@10 vs exact `0.9041666666666663`.
- the new native path therefore improves the measured HNSW query row by
  `7.90x` on this bounded artifact, but its candidate set is not byte-for-byte
  identical to the Python HNSW traversal row.
- on `docs10000_q128_c32768`, building the missing sidecar took
  `46.472632209995936s` for `32768` centroids, `558040` graph edges, and
  `3543200` graph bytes.
- on that `docs10000_q128_c32768` slice, native HNSW+PQ sparse rerank reached
  `69.01278987944805` QPS, with MRR@10 `0.9895833333333334`, candidate
  recall@10 vs exact `0.9484375000000002`, and final recall@10 vs exact
  `0.8867187500000006`.
- compared with native exact-centroid TAC+PQ on the same 10k row
  (`55.84355601133033` QPS and final recall@10 `0.8882812500000007`), the HNSW
  sparse rerank path is faster but gives up a small amount of exact top-10
  overlap. This is a measured speed/quality tradeoff, not a free replacement.

Native streaming HNSW/PQ materialization pass:
- the first native HNSW+PQ streaming bridge copied the memmap arrays through
  Python lists into Mojo lists. That is fast at bounded scale but does not
  materialize cleanly toward the paper's multi-million-centroid shape.
- `streaming_tac_hnsw_pq_mojo_address` now prepares an address-backed native
  engine over the same streaming memmaps and HNSW sidecar. The Python wrapper
  keeps the memmap-owning `base_index` and `graph` objects alive while Mojo
  reads their addresses.
- the address-backed engine uses the same sparse HNSW candidate traversal and
  residual-PQ rerank semantics, but avoids Python-list copies for centroids,
  postings, token-centroid positions, residual norms, PQ codes/codebooks, and
  graph arrays.
- on `docs1500_q48_c32768`, the address-backed engine reached `103.826705`
  QPS with candidate recall@10 vs exact `0.94375` and final recall@10 vs exact
  `0.9041666666666663`. The earlier scalar pointer-dot attempt was only
  `42.174949` QPS, so the address engine now uses a SIMD pointer dot for
  centroid scores.
- on `docs10000_q128_c32768`, address-backed preparation took
  `0.01598953700158745s` versus `2.0018705330003286s` for the List-backed
  native prepare. Query QPS was lower: `42.237957` QPS versus about `69` QPS
  for the List-backed native HNSW+PQ row, with the same candidate/final recall.
- interpretation: keep `streaming_tac_hnsw_pq_mojo` as the faster bounded
  query engine, and use `streaming_tac_hnsw_pq_mojo_address` when
  materialization pressure matters more than peak local QPS.

Larger native streaming scale rows:

| Slice | Docs | Doc vectors | Queries | Query vectors | Positives | MRR@10 | Exact MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact | Python TAC+PQ QPS | Mojo TAC+PQ QPS | Exact MaxSim QPS | Index bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `docs2500_q64_c32768` | `2570` | `188537` | `64` | `2048` | `70` | `1.0` | `1.0` | `0.9390624999999997` | `0.8890624999999996` | `36.191123849798835` | `73.48328381526854` | `27.766588902990385` | `25548967` |
| `docs5000_q96_c32768` | `5103` | `371673` | `96` | `3072` | `103` | `1.0` | `0.9947916666666666` | `0.9385416666666666` | `0.8791666666666665` | `26.36500059866449` | `58.04598536948425` | `13.883993382776076` | `33429239` |
| `docs10000_q128_c32768` | `10135` | `739372` | `128` | `4096` | `135` | `0.9895833333333334` | `0.9893973214285714` | `0.9570312500000002` | `0.8882812500000007` | `19.94036997697634` | `55.84355601133033` | `6.8977521982513945` | `49263251` |

Interpretation:
- native streaming TAC+PQ stayed faster than exact MaxSim as document vectors
  increased from `188537` to `739372`; the measured speedup versus exact
  MaxSim rose from `2.65x` to `8.10x`.
- judged MRR stayed matched or slightly above exact on these selected
  judged-positive slices, but final top-10 overlap stayed approximate around
  `0.88` to `0.89`.
- exact centroid scan with `32768` centroids is still viable at this bounded
  scale. It is not proof that exact centroid scan will survive the paper's
  about `4M` centroid setting.
- the `docs10000_q128` run made the wall-clock bottleneck visible: the quiet
  CPU ColBERT snapshot materialization phase dominated user-visible waiting
  time before query benchmarking. The scale runner now emits per-stage progress
  and timing for future runs.

## USL Batch-Cap Sweep

Implemented:
- `python/kayak_bridge/usl_scaling.py` fits Universal Scalability Law
  parameters from measured throughput points using a linearized least-squares
  fit. Reason: the fit is auditable and does not add a SciPy dependency.
- `python/scripts/sweep_tachiom_streaming_usl.py` sweeps
  `max_query_batch_size` for a fixed streaming artifact and fits USL over
  median QPS across repeated measurements.
- `python/kayak_bridge/tachiom_streaming_benchmark.py`,
  `python/scripts/bench_tachiom_streaming_index.py`, and
  `python/scripts/run_tachiom_streaming_scale_gate.py` now accept
  `max_query_batch_size` so a measured batch cap can be applied explicitly.

Validation:

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_usl_scaling.py \
  python/tests/test_tachiom_streaming_index.py
```

Result: `5` tests passed.

Bounded slice command:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 60 --force -- \
  pixi run env PYTHONPATH=python python \
  python/scripts/sweep_tachiom_streaming_usl.py \
  --snapshot .cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/snapshot \
  --index .cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/streaming_tachiom_index_c32768 \
  --engine streaming_tac_hnsw_pq_mojo \
  --batch-sizes 1,2,4,8,16,32,48 \
  --warmup-iterations 1 \
  --measurement-iterations 3 \
  --sweep-repeats 3 \
  --output .cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/streaming_tachiom_index_c32768/usl_hnsw_pq_mojo_batch_sweep.json \
  --emit-quiet-mean
```

Quiet wrapper log: `.cache/kayak/bench_quiet/20260501T212916Z`

Result:

| Batch cap | Median batch s | Median QPS |
| ---: | ---: | ---: |
| `1` | `0.37981928499599843` | `126.37588952468725` |
| `2` | `0.37966733233285294` | `126.42646841661525` |
| `4` | `0.3784242023320985` | `126.84178153562185` |
| `8` | `0.3792684686656382` | `126.5594268062306` |
| `16` | `0.37912945966430317` | `126.60583021562391` |
| `32` | `0.38030276566375204` | `126.21522727089398` |
| `48` | `0.3792783400009891` | `126.5561328914138` |

USL fit:
- `alpha`: `1.0184967561686984`
- `beta`: `-0.00007661380105196282`
- `r_squared`: `-18.881725566866045`
- best measured median cap: `4`

Larger slice command:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 60 --force -- \
  pixi run env PYTHONPATH=python python \
  python/scripts/sweep_tachiom_streaming_usl.py \
  --snapshot .cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/snapshot \
  --index .cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/streaming_tachiom_index_c32768 \
  --engine streaming_tac_hnsw_pq_mojo \
  --batch-sizes 1,2,4,8,16,32,64,128 \
  --warmup-iterations 1 \
  --measurement-iterations 3 \
  --sweep-repeats 3 \
  --output .cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/streaming_tachiom_index_c32768/usl_hnsw_pq_mojo_batch_sweep.json \
  --emit-quiet-mean
```

Quiet wrapper log: `.cache/kayak/bench_quiet/20260501T213019Z`

Result:

| Batch cap | Median batch s | Median QPS |
| ---: | ---: | ---: |
| `1` | `1.8425459063316036` | `69.46909684049076` |
| `2` | `1.8299118306676974` | `69.94872531825504` |
| `4` | `1.8395472576642835` | `69.58233851655685` |
| `8` | `1.8397520496655488` | `69.57459295847465` |
| `16` | `1.8375751576653176` | `69.65701482524774` |
| `32` | `1.833400016335266` | `69.81564244547994` |
| `64` | `1.825026084331815` | `70.13598386286289` |
| `128` | `1.83946473733522` | `69.58546005368385` |

USL fit:
- `alpha`: `0.9315768813994619`
- `beta`: `0.00008892833648842428`
- `r_squared`: `-51.451364503489344`
- best measured median cap: `64`

Interpretation:
- The sweep did not reveal a material same-shape batching optimization. On the
  `docs1500_q48` slice, best measured QPS is only about `0.37%` above batch
  cap `1`; on the `docs10000_q128` slice, best measured QPS is about `0.96%`
  above batch cap `1`.
- Negative `r_squared` means the USL curve explains these points worse than a
  flat mean. That is useful evidence: current query time is dominated by
  per-query HNSW/PQ work inside the native path, not by Python/native call
  batching overhead.
- The only sound use of this sweep today is to pass the best measured cap
  explicitly for a fixed artifact (`4` on the bounded 1.5k slice, `64` on the
  bounded 10k slice). It is not evidence for a global default, and it is not a
  route to paper-scale throughput by itself.

## Decision

The implemented TAC first gate is useful enough to keep:
- it preserves exact final recall at reachable candidate budgets on the tested
  token-structured workloads
- the Mojo path makes TAC faster than exact MaxSim at moderate and large vector
  counts
- it gives Kayak a high-recall candidate lane that complements the lower-memory
  i8 lane

It is not the full Tachiom paper result:
- the full paper-shaped TAC + HNSW + PQ Python reference exists, but it has not
  been run on paper-scale judged artifacts
- the HNSW centroid layer now has a native streaming HNSW+PQ sparse rerank
  query path, but graph construction is still the Python reference and full
  paper-scale corpus evidence is still missing
- residual PQ now has a native streaming reader that beats exact MaxSim on the
  bounded `docs1500_q48_c32768` slice, but it has not been validated on
  full-corpus MS MARCO or LoTTE
- exact-rerank TAC memory is still exact vectors plus TAC sidecar
- evidence now includes bounded MS MARCO dev.small token-id gates and a
  judged-positive streaming TAC/PQ gate, but not full-corpus MS MARCO/LoTTE
  paper-scale evaluation

Next sound gate:
- scale the judged-positive streaming snapshot beyond `1036` documents and
  `74804` vectors while keeping the required positives included
- benchmark the persisted centroid HNSW sidecar at larger centroid/document
  counts to identify the scale where graph traversal becomes useful, or judge
  the paper-throughput target out of scope for the current Python/kANNolo-free
  implementation
- profile the HNSW traversal path separately; the exact-centroid reader's
  largest Python top-k/posting overhead has now been reduced on the measured
  `docs1500_q48_c32768` gate
- test whether a stronger residual-PQ refine policy can close the recall gap
  without giving back the byte savings
- move HNSW graph construction and neighbor selection out of the Python
  reference path if centroid traversal is kept as a production candidate
  family
- run the same token-id-bearing pipeline on larger bounded MS MARCO and LoTTE
  slices while keeping document/query vector counts explicit
