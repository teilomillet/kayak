# 2026-04-27: GPU I8 Posting Accumulation Probe

## Question

Can a fused GPU primitive accumulate selected centroid posting scores into
per-document scores faster than the CPU posting-accumulation substep?

## Change

Added a benchmark-only dense accumulation probe:

- `python/kayak_bridge/gpu_i8_candidate_posting_accumulation.py`
- `python/scripts/profile_gpu_i8_candidate_posting_accumulation.py`
- `profile_gpu_i8_candidate_posting_accumulation_raw`
- `profile_gpu_i8_candidate_posting_accumulation`

The first kernel used one GPU thread per `(query, document)` score. Each thread
scanned the selected centroids for every query vector, binary-searched the
sorted posting lists for its document id, kept the max selected centroid score
for that query vector, and summed those maxima across query vectors.

After that variant lost to isolated CPU posting accumulation, the active probe
was changed to a posting-oriented atomic variant:

- reset dense `[query_count, query_vector_count, document_count]` best-score
  storage
- traverse posting lists by selected centroid
- use `std.os.atomic.Atomic.compare_exchange` to perform `Float32` atomic max
  into per-query-vector document best scores
- reduce query-vector best scores into dense `[query_count, document_count]`
  document scores

Reason: the first variant validated the same per-query-vector max-then-sum
semantics as the CPU candidate generator without atomics. The second variant
then tested whether local Mojo GPU atomics were a viable posting-oriented
replacement. Neither was assumed to be the fastest GPU shape.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_candidate_posting_accumulation
```

Artifacts:

- report: `.cache/kayak/gpu_i8_candidate_posting_accumulation/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T131446Z`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

All rows validated:

- `score_mismatch_count = 0`
- `score_delta_max_abs = 0.0`
- `selected_position_out_of_range_count = 0`
- `doc_index_out_of_range_count = 0`

Non-full candidate-generation rows:

| case | expanded postings | CPU candidate s | CPU posting s | GPU kernel s | GPU all measured s | GPU all / CPU candidate | GPU all / CPU posting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `16087` | `0.0004048246670815085` | `0.00006184893766666666` | `0.0005130987222222222` | `0.0005291446220625515` | `1.307095799960263` | `8.555435906019332` |
| `doc_vectors64` | `53949` | `0.0003647606663434999` | `0.00007306479333333333` | `0.001267959414893617` | `0.001289314852360701` | `3.534687183476456` | `17.646184893436153` |
| `query_batch4` | `15686` | `0.0005829576663624417` | `0.000053756344886495236` | `0.0004879311306122449` | `0.00050432455093315` | `0.8651135065776745` | `9.38167488875992` |

Summary:

- non-full all-measured GPU accumulation ranged from about `0.865x` to `3.535x`
  of full CPU candidate generation
- non-full all-measured GPU accumulation ranged from about `8.56x` to `17.65x`
  of isolated CPU posting accumulation
- the atomic variant preserved exact scores but was slower than the earlier
  document-centric variant on these rows

The two full-window rows are reported but are not optimization targets because
candidate generation is intentionally near-zero when `candidate_k` equals
`document_count`.

## Interpretation

This validates the dense accumulation score semantics on GPU. The CPU reference
and GPU output matched exactly on the measured synthetic rows for both the
document-centric variant and the posting-oriented atomic variant.

It also falsifies both obvious work assignments as likely final
candidate-generation primitives. The document-centric binary-search kernel
repeatedly searches posting lists for every document. The posting-oriented
atomic variant avoids repeated searches, but global `Float32` compare-exchange
contention and extra reset/reduce launches dominate the measured rows.

The next GPU candidate-generation experiment should keep the good part of this
probe, reducing on device before readback, while avoiding both repeated posting
search and high-contention global atomics. A two-stage segmented or tiled
reduction is now more plausible than another global atomic update variant.

## Next Step

Design the smallest tiled/segmented reduction probe that preserves CPU
max-then-sum semantics without global atomics.

Reason: we now know that plain traversal readback is too shallow and
document-centric dense accumulation is too much repeated work. We also know
that naive posting-oriented global atomic max is worse on the measured shapes.
