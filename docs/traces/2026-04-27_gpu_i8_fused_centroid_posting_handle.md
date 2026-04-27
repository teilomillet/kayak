# 2026-04-27: Prepared GPU I8 Fused Centroid-Posting Handle

## Question

Does the fused GPU i8 centroid-scoring, centroid-selection, posting-accumulation,
and top-k primitive still win when it is exposed through an explicit
prepare/score/release handle?

## Change

Added a benchmark-only prepared handle:

- `python/kayak_bridge/gpu_i8_fused_centroid_posting_handle.py`
- `python/scripts/profile_gpu_i8_fused_centroid_posting_handle.py`
- `profile_gpu_i8_fused_centroid_posting_handle_raw`
- `profile_gpu_i8_fused_centroid_posting_handle`

The Mojo bridge now exposes an internal fused session handle that:

- prepares token codes, token scales, centroid token indices, centroid posting
  offsets, and centroid posting document ids once
- scores one query batch against the prepared payload
- selects `centroids_per_query_vector` centroids per query vector on device
- accumulates selected posting scores into dense document scores on device
- reads document scores back
- returns host top-k positions and scores
- releases the prepared payload explicitly

Reason: the previous fused probe showed a resident-payload win, but still
measured residency as an in-call benchmark projection. This change tests the
real ownership boundary needed by a serving path without introducing a hidden
global cache or public GPU search API.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_fused_centroid_posting_handle
```

Artifacts:

- report: `.cache/kayak/gpu_i8_fused_centroid_posting_handle/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T144646Z`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`
- driver: `595.58.03`

All rows validated:

- minimum top-k position agreement: `1.0`
- maximum top-k position mismatch: `0`
- maximum top-k score delta: `0.00006103515625`

Non-full rows:

| case | cpqv | expanded postings | CPU candidate s | CPU slice s | prepare extension s | score extension s | score host+extension s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `4` | `16087` | `0.0003938443345153549` | `0.00022360219208901905` | `0.0005481430016516242` | `0.00009306399927784999` | `0.00009422633229405619` |
| `doc_vectors64` | `16` | `53949` | `0.0003579213334887754` | `0.0001802921357603447` | `0.001834843002143316` | `0.00011383233262070765` | `0.000114987666165689` |
| `query_batch4` | `8` | `15686` | `0.0005702073334153587` | `0.00024137733043539875` | `0.0005292679998092353` | `0.00013805400036896268` | `0.0001394066687983771` |

Ratios:

| case | score extension / CPU candidate | score extension / CPU slice |
| --- | ---: | ---: |
| `query_vectors32` | `0.23629640221274195` | `0.41620342988766384` |
| `doc_vectors64` | `0.31803729470704345` | `0.6313771376696127` |
| `query_batch4` | `0.24211193416623314` | `0.571942692878157` |

Summary:

- non-full score-call path ranged from about `0.236x` to `0.318x` of full CPU
  candidate generation
- non-full score-call path ranged from about `0.416x` to `0.631x` of the CPU
  centroid-selection/posting/top-k slice
- score-call host marshalling added about `1us` to `2us`, so the Python
  boundary is not the current dominant cost for these rows
- prepare cost is explicit setup cost, not hidden fallback or implicit cache

## Interpretation

The prepared handle preserves correctness on the wide matrix and keeps the
non-full serving-shaped score call faster than both full CPU candidate
generation and the narrower CPU centroid-selection/posting/top-k slice.

This result is weaker than the previous in-call resident envelope
(`0.200x` to `0.282x` of CPU candidate generation) because it includes the real
cross-call extension boundary and Python result construction. That difference
is useful evidence, not a regression to hide.

The full-window rows remain explanatory only because CPU candidate generation
shortcuts them when `candidate_k == document_count`. The first full-window
prepare call also includes a large setup outlier, so prepare timing should be
studied separately before using it for amortization claims.

## Next Step

Profile the prepared handle's score call by substep: query H2D, centroid
scoring, device heap selection, selected-posting accumulation, reduction,
document-score readback, and host top-k.

Reason: the handle is now correct and faster on the target non-full rows, but
the next optimization should be selected by measured remaining cost rather than
by guessing whether kernels, readback, or host top-k dominate.
