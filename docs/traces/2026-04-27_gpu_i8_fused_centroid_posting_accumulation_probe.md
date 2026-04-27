# 2026-04-27: GPU I8 Fused Centroid-Posting Accumulation Probe

## Question

Can GPU centroid scoring, centroid selection, and selected-posting
accumulation be fused into one internal primitive that beats the measured CPU
candidate-generation substeps?

## Change

Added a benchmark-only fused probe:

- `python/kayak_bridge/gpu_i8_fused_centroid_posting_accumulation.py`
- `python/scripts/profile_gpu_i8_fused_centroid_posting_accumulation.py`
- `profile_gpu_i8_fused_centroid_posting_accumulation_raw`
- `profile_gpu_i8_fused_centroid_posting_accumulation`

The Mojo bridge now has a fused typed-address profile function that:

- copies query values and i8 candidate-generation payload tensors
- scores sampled centroids on GPU
- selects centroids on device
- accumulates selected centroid posting scores into dense document scores
- reads document scores back
- runs host document top-k
- validates selected centroids, document scores, and top-k positions against
  the CPU i8 references

Reason: separate probes showed GPU centroid scoring/selection and GPU posting
accumulation could each help, but the split path still paid centroid-score
readback and selected-centroid upload. This probe tests the combined boundary
directly instead of assuming the separate wins compose.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_fused_centroid_posting_accumulation
```

Artifacts:

- report: `.cache/kayak/gpu_i8_fused_centroid_posting_accumulation/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T142642Z`
- repeated-scan selector comparison log:
  `.cache/kayak/bench_quiet/20260427T142357Z`
- heap-selector smoke:
  `.cache/kayak/gpu_i8_fused_centroid_posting_accumulation/heap_smoke.json`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

All rows validated:

- `selected_position_mismatch_count = 0`
- `selected_score_mismatch_count = 0`
- `score_mismatch_count = 0`
- `topk_position_mismatch_count = 0`
- `centroid_token_out_of_range_count = 0`
- `doc_index_out_of_range_count = 0`

Non-full rows:

| case | cpqv | expanded postings | CPU candidate s | CPU slice s | centroid score kernel s | device selection kernel s | accumulation kernel s | host top-k s | resident s | cold s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `4` | `16087` | `0.0004123660013040838` | `0.00023295086212494307` | `0.000016459594101508917` | `0.00001472401275447633` | `0.00000996234111101885` | `0.000032428982` | `0.00008228963049290188` | `0.00016319201039975483` |
| `doc_vectors64` | `16` | `53949` | `0.0003564716668430871` | `0.00018564402082978478` | `0.000010623524657290369` | `0.000027793167631396157` | `0.000021869846966660594` | `0.000032471838666666667` | `0.00010042810311377398` | `0.00029612457379781307` |
| `query_batch4` | `8` | `15686` | `0.0005854526668069108` | `0.0002509464846687582` | `0.000010612454934474459` | `0.00001906669685414681` | `0.000012163834786134199` | `0.0000651901281779661` | `0.00011752920430931093` | `0.00018763347614274067` |

Ratios:

| case | resident / CPU candidate | cold / CPU candidate | resident / CPU slice |
| --- | ---: | ---: | ---: |
| `query_vectors32` | `0.19955483777194447` | `0.3957455510000084` | `0.35324887721929044` |
| `doc_vectors64` | `0.28172814968203563` | `0.8307099871927891` | `0.5409713852613414` |
| `query_batch4` | `0.2007492850793922` | `0.32049299077601506` | `0.4683436967226138` |

Summary:

- non-full resident fused path ranged from about `0.200x` to `0.282x` of
  full CPU candidate generation
- non-full cold fused path ranged from about `0.320x` to `0.831x` of full CPU
  candidate generation
- non-full resident fused path ranged from about `0.353x` to `0.541x` of the
  CPU centroid-selection/posting/top-k slice
- selected centroid positions matched exactly on all rows
- top-k positions matched exactly on all rows

## Rejected Selector

The first fused selector used one GPU lane per query vector and repeatedly
rescanned all centroids for each selected rank. It was correct, but too slow.
On the same non-full rows, the repeated-scan selection kernel was about `55us`,
`596us`, and `172us`; the heap selector reduced those to about `15us`, `28us`,
and `19us`.

Reason: `centroids_per_query_vector` is explicit and shape-dependent. A
repeated scan has cost proportional to `centroid_count * cpqv`, while a bounded
worst-first heap has cost closer to `centroid_count * log(cpqv)` and preserves
the CPU tie order.

## Interpretation

This is the first candidate-generation-shaped GPU primitive that wins on every
measured non-full wide row while preserving selected-centroid, document-score,
and top-k agreement against the CPU i8 reference.

It is still not a public GPU backend. The benchmark copies the payload inside
the profiling call, and host top-k remains in the path. The resident result is
the relevant target for an explicit prepared GPU payload; the cold result shows
why hidden copies would erase part of the win, especially on `doc_vectors64`.

The full-window rows remain explanatory only because CPU candidate generation
shortcuts them when `candidate_k == document_count`.

## Next Step

Build an explicit prepared fused payload boundary and compare that scoped
pipeline against the current FastPlaid matrix.

Reason: the kernel chain is now fast enough that ownership and residency are
the next correctness/performance boundary. Public API work should still wait
until the fused primitive is measured through a reusable internal handle.
