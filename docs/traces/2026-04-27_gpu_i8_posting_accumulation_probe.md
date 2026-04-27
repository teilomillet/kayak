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

The surviving qv-doc reduce probe now also times host candidate top-k after
score readback and compares the resulting top-k positions against the CPU i8
score reference.

Reason: the primitive is only useful as candidate generation if score
accumulation plus top-k is competitive. Measuring accumulation without top-k
would overstate the end-to-end value.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_candidate_posting_accumulation
```

Artifacts:

- report: `.cache/kayak/gpu_i8_candidate_posting_accumulation/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T133911Z`
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
- `topk_position_mismatch_count = 0`
- `selected_position_out_of_range_count = 0`
- `doc_index_out_of_range_count = 0`

Non-full candidate-generation rows:

| case | expanded postings | CPU candidate s | CPU centroid scoring + selection s | CPU posting s | GPU kernel s | GPU all measured s | host top-k s | GPU all + top-k s | projected resident s | projected cold s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `16087` | `0.00039362766680521116` | `0.00009919261833513072` | `0.00006956520333333333` | `0.000009955043608025203` | `0.000025875730836370773` | `0.000033194848` | `0.00005907057883637078` | `0.00015105268846196017` | `0.0001582631971715015` |
| `doc_vectors64` | `53949` | `0.0003659856671826371` | `0.00004353978041700765` | `0.00007368735733333333` | `0.000021899478956074252` | `0.00004328307774549517` | `0.000033133537` | `0.00007641661474549517` | `0.00010727945747798151` | `0.00011995639516250282` |
| `query_batch4` | `15686` | `0.0005803493337831848` | `0.00006183196065008171` | `0.00005415339284291359` | `0.000012187922732818217` | `0.00002844377957147834` | `0.00006646704547975596` | `0.00009491082505123431` | `0.00014941896135796205` | `0.00015674278570131602` |

Non-full ratios:

| case | GPU all / CPU candidate | GPU all + top-k / CPU candidate | projected resident / CPU candidate | projected cold / CPU candidate |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.06573657992157913` | `0.1500671416615697` | `0.3837451002566581` | `0.40206319453103617` |
| `doc_vectors64` | `0.11826440657823821` | `0.2087967415056206` | `0.2931247507691225` | `0.3277625489706438` |
| `query_batch4` | `0.04901147966528859` | `0.16354085294201864` | `0.25746382852536215` | `0.27008351104590783` |

Summary:

- non-full all-measured GPU accumulation ranged from about `0.049x` to `0.118x`
  of full CPU candidate generation
- non-full all-measured plus host top-k ranged from about `0.150x` to `0.209x`
  of full CPU candidate generation
- non-full all-measured GPU accumulation ranged from about `0.372x` to `0.587x`
  of isolated CPU posting accumulation
- the resident-payload projection ranged from about `0.257x` to `0.384x` of
  full CPU candidate generation
- the cold-payload projection ranged from about `0.270x` to `0.402x` of full
  CPU candidate generation
- the race-free qv-doc reduce variant preserved exact scores and exact top-k
  order while beating isolated CPU posting accumulation on all three non-full
  rows

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
agreement and exact top-k order against the CPU i8 reference.

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

The post-top-k envelope changes the next optimization choice. Host top-k is
visible at about `33us` on the two-query rows and about `66us` on the four-query
row, so GPU top-k is a real candidate. However, the projected resident envelope
also includes CPU centroid scoring plus selection at about `44us` to `99us` on
the same non-full rows. That means optimizing only the qv-doc reduce kernel is
not obviously the highest-leverage next step.

## Next Step

Design the next probe around the remaining measured envelope, not around a
kernel preference:

- GPU or fused host/GPU top-k if readback plus host selection remains exposed
  after resident payloads
- GPU selected-centroid scoring/selection if CPU selection stays comparable to
  the GPU accumulation path
- tiled or segmented accumulation if dense intermediate memory becomes the
  limiter on larger explicit vector-count shapes

Reason: the qv-doc reduce path is good enough to keep, but the evidence does
not support spending all effort on one kernel. The next optimization should
attack the largest measured slice while preserving the exact agreement
contract.
