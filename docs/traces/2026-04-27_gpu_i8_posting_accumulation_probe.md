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

After that variant lost to isolated CPU posting accumulation, the probe tested
three smaller work assignments:

- posting-oriented global `Float32` atomic max into dense
  `[query_count, query_vector_count, document_count]` best-score storage
- race-free qv-doc reduce, with one lane per
  `[query, query_vector, document]` contribution followed by an on-device reduce
  to `[query_count, document_count]`
- qv-doc `Float32` atomic add directly into `[query_count, document_count]`
  scores

Reason: the first variant validated CPU max-then-sum semantics. The follow-up
variants isolate the important design question: whether the GPU should traverse
posting lists, search posting lists by document, or use low-contention atomics.
No variant was treated as correct or fast until it passed the benchmark
agreement checks.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_candidate_posting_accumulation
```

Artifacts:

- report: `.cache/kayak/gpu_i8_candidate_posting_accumulation/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T132611Z`
- rejected qv-doc atomic-add report:
  `.cache/kayak/gpu_i8_candidate_posting_accumulation/qv_doc_atomic_add_failed_summary.json`

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
| `query_vectors32` | `16087` | `0.00041145766590489075` | `0.000060911268` | `0.000010049280806107953` | `0.000026147748014157988` | `0.0635490602821873` | `0.42927604157178256` |
| `doc_vectors64` | `53949` | `0.0003577679999580141` | `0.00006988843366666667` | `0.00002188809049443532` | `0.00004342075033113921` | `0.12136566248584237` | `0.6212866429119696` |
| `query_batch4` | `15686` | `0.0005763286668904281` | `0.00005463696603515483` | `0.00001224410436866761` | `0.000028612139514269478` | `0.0496455254753261` | `0.5236773120941542` |

Summary:

- non-full all-measured GPU accumulation ranged from about `0.050x` to `0.121x`
  of full CPU candidate generation
- non-full all-measured GPU accumulation ranged from about `0.429x` to `0.621x`
  of isolated CPU posting accumulation
- the race-free qv-doc reduce variant preserved exact scores and beat isolated
  CPU posting accumulation on all three non-full rows

The two full-window rows are reported but are not optimization targets because
candidate generation is intentionally near-zero when `candidate_k` equals
`document_count`.

Rejected qv-doc atomic add:

- command: `pixi run profile_gpu_i8_candidate_posting_accumulation`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T132356Z`
- status: `blocked_gpu_candidate_posting_accumulation_failed`
- reason: all wide rows failed the exact score-agreement contract
- max absolute score delta: `9.1552734375e-05`
- score mismatches ranged from `303` to `786` across the wide rows
- non-full all-measured timing was also worse than qv-doc reduce:
  `0.00005665046160366033`, `0.000047955148648906834`, and
  `0.000043898084318970214` seconds

## Interpretation

This validates a GPU accumulation primitive that is faster than isolated CPU
posting accumulation on the measured standard rows, while preserving exact score
agreement against the CPU i8 reference.

The important result is not "GPU wins everywhere." The full-window rows still
show why end-to-end claims need care: when `candidate_k == document_count`, CPU
candidate generation is intentionally near-zero and this GPU primitive is not
the relevant comparison. The non-full rows are the target for this primitive.

The rejected variants matter:

- document-centric scoring is deterministic but under-parallelized
- posting-oriented global atomic max has too much contention
- qv-doc atomic add has lower contention but violates exact score agreement
  because floating-point addition order is no longer deterministic

The current best primitive is race-free qv-doc reduce. It pays extra memory for
`[query_count, query_vector_count, document_count]`, but that buys deterministic
scores and avoids global atomic contention.

## Next Step

Design the smallest tiled or segmented reduction probe that preserves CPU
max-then-sum semantics, reduces the dense intermediate, and avoids global
atomics.

Reason: the qv-doc reduce path is now good enough to keep, but the dense
intermediate scales with `query_count * query_vector_count * document_count`.
The next real optimization is to reduce that memory traffic and allocation
surface without losing deterministic agreement.
