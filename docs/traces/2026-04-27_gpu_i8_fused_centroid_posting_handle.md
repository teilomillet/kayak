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
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T145047Z`

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
| `query_vectors32` | `4` | `16087` | `0.00039849666548737633` | `0.00022449504242612897` | `0.000547851999726845` | `0.00009333466732641682` | `0.00009446666808798909` |
| `doc_vectors64` | `16` | `53949` | `0.00036635366632253863` | `0.00018070177804926525` | `0.0018424569971102756` | `0.00011298733321988645` | `0.00011410633427052139` |
| `query_batch4` | `8` | `15686` | `0.0005736203347623814` | `0.00024253767028910035` | `0.0005445160022645723` | `0.0001347946672467515` | `0.00013597333357514194` |

Ratios:

| case | score extension / CPU candidate | score extension / CPU slice |
| --- | ---: | ---: |
| `query_vectors32` | `0.2342169343180451` | `0.4157538015884203` |
| `doc_vectors64` | `0.3084105431618968` | `0.6252696262295903` |
| `query_batch4` | `0.2349893458755944` | `0.5557679641520378` |

Summary:

- non-full score-call path ranged from about `0.234x` to `0.308x` of full CPU
  candidate generation
- non-full score-call path ranged from about `0.416x` to `0.625x` of the CPU
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
