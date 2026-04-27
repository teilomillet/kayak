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

The same probe now also times a deliberately simple GPU document top-k variant.
That kernel assigns one GPU lane per query and repeatedly scans the dense
document-score row to choose each top-k rank.

Reason: host top-k was a visible measured slice, so it deserved a direct GPU
test. The one-lane design was chosen as a correctness-preserving lower bound,
not as the assumed final GPU top-k design.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_candidate_posting_accumulation
```

Artifacts:

- report: `.cache/kayak/gpu_i8_candidate_posting_accumulation/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T140440Z`
- rejected qv-doc atomic-add report:
  `.cache/kayak/gpu_i8_candidate_posting_accumulation/qv_doc_atomic_add_failed_summary.json`
- device top-k smoke report:
  `.cache/kayak/gpu_i8_candidate_posting_accumulation/device_topk_smoke.json`

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
- `device_topk_position_mismatch_count = 0`
- `selected_position_out_of_range_count = 0`
- `doc_index_out_of_range_count = 0`

Non-full candidate-generation rows:

| case | expanded postings | CPU candidate s | CPU centroid scoring + selection s | CPU posting s | GPU kernel s | GPU all measured s | host top-k s | GPU all + top-k s | projected resident s | projected cold s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `16087` | `0.00039813933411399677` | `0.00010302801410820132` | `0.000064448824` | `0.00000996820922132847` | `0.0000259893285237152` | `0.000032067768333333334` | `0.000058057096857048535` | `0.00015374020210247595` | `0.00016108511096524984` |
| `doc_vectors64` | `53949` | `0.0003527180003099299` | `0.00004527691943583493` | `0.00007054474699999999` | `0.00002189056996711728` | `0.00004313827101595736` | `0.00003230754233333333` | `0.00007544581334929069` | `0.00010809420059175429` | `0.00012072273278512562` |
| `query_batch4` | `15686` | `0.000572260999736803` | `0.00006180235337715202` | `0.00005397287991302342` | `0.00001218611320754717` | `0.00002846588382270652` | `0.00006497294855305467` | `0.0000934388323757612` | `0.00014787015433671036` | `0.00015524118575291321` |

Non-full ratios:

| case | GPU all / CPU candidate | GPU all + top-k / CPU candidate | projected resident / CPU candidate | projected cold / CPU candidate |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.06527696787746633` | `0.14582105278858332` | `0.3861467303766989` | `0.40459481684652476` |
| `doc_vectors64` | `0.12230243701215186` | `0.21389839271882122` | `0.3064606867150896` | `0.3422641676326349` |
| `query_batch4` | `0.04974283383945208` | `0.1632800984493719` | `0.25839635132346866` | `0.27127689257928195` |

Device top-k follow-up:

| case | host top-k s | device top-k kernel s | device top-k D2H s | GPU all + host top-k / CPU candidate | GPU all + device top-k / CPU candidate | projected host resident / CPU candidate | projected device resident / CPU candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.000032067768333333334` | `0.0010210662478632479` | `0.0000041253007136975025` | `0.14582105278858332` | `2.631397545699474` | `0.3861467303766989` | `2.8717232232875896` |
| `doc_vectors64` | `0.00003230754233333333` | `0.0010203801794871794` | `0.000004132419770526823` | `0.21389839271882122` | `3.0169254612427725` | `0.3064606867150896` | `3.1094877552390408` |
| `query_batch4` | `0.00006497294855305467` | `0.0010211896324786324` | `0.00000412991210870313` | `0.1632800984493719` | `1.8350227724057404` | `0.25839635132346866` | `1.9301390252798372` |

Summary:

- non-full all-measured GPU accumulation ranged from about `0.050x` to `0.122x`
  of full CPU candidate generation
- non-full all-measured plus host top-k ranged from about `0.146x` to `0.214x`
  of full CPU candidate generation
- non-full all-measured plus device top-k ranged from about `1.835x` to
  `3.017x` of full CPU candidate generation
- non-full all-measured GPU accumulation ranged from about `0.403x` to `0.612x`
  of isolated CPU posting accumulation
- the host top-k resident-payload projection ranged from about `0.258x` to
  `0.386x` of full CPU candidate generation
- the host top-k cold-payload projection ranged from about `0.271x` to `0.405x`
  of full CPU candidate generation
- the device top-k resident-payload projection ranged from about `1.930x` to
  `3.109x` of full CPU candidate generation
- the device top-k cold-payload projection ranged from about `1.943x` to
  `3.145x` of full CPU candidate generation
- the one-lane device top-k preserved exact top-k positions but was rejected by
  timing
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
- one-lane device document top-k is deterministic and exact, but serializes too
  much work per query; it is slower than host top-k and slower than full CPU
  candidate generation on the non-full rows

The current best primitive is race-free qv-doc reduce. It pays extra memory for
`[query_count, query_vector_count, document_count]`, but that buys deterministic
scores and avoids global atomic contention.

The post-top-k envelope changes the next optimization choice. Host top-k is
visible at about `33us` on the two-query rows and about `66us` on the four-query
row, but this probe falsifies the simplest GPU top-k shape. A future GPU top-k
would need a parallel segmented or heap-like design, not one lane per query.
The projected resident envelope also includes CPU centroid scoring plus
selection at about `45us` to `103us` on the same non-full rows. That means
optimizing only the qv-doc reduce kernel or only document top-k is not
obviously the highest-leverage next step.

## Next Step

Design the next probe around the remaining measured envelope, not around a
kernel preference:

- parallel GPU or fused host/GPU top-k only if readback plus host selection
  remains exposed after resident payloads
- GPU selected-centroid scoring/selection if CPU selection stays comparable to
  the GPU accumulation path
- tiled or segmented accumulation if dense intermediate memory becomes the
  limiter on larger explicit vector-count shapes

Reason: the qv-doc reduce path is good enough to keep, but the evidence does
not support spending all effort on one kernel. The next optimization should
attack the largest measured slice while preserving the exact agreement
contract.
