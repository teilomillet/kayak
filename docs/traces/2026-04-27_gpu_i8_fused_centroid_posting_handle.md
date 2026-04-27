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
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T150112Z`

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
| `query_vectors32` | `4` | `16087` | `0.00041524466723785736` | `0.00023471612817232913` | `0.0005263919993012678` | `0.00009281333283676456` | `0.00009400866595872989` |
| `doc_vectors64` | `16` | `53949` | `0.00036512799973327975` | `0.00018144819185647977` | `0.0018464940003468655` | `0.00011206566826634419` | `0.00011310433546896093` |
| `query_batch4` | `8` | `15686` | `0.0005804433349112514` | `0.0002498248270931103` | `0.0005343660013750196` | `0.00013698866678168997` | `0.00013820433377986774` |

Ratios:

| case | score extension / CPU candidate | score extension / CPU slice |
| --- | ---: | ---: |
| `query_vectors32` | `0.22351480984486652` | `0.3954280157885903` |
| `doc_vectors64` | `0.30692159557252907` | `0.6176179939835656` |
| `query_batch4` | `0.23600695975367733` | `0.5483388835915574` |

Summary:

- non-full score-call path ranged from about `0.224x` to `0.307x` of full CPU
  candidate generation
- non-full score-call path ranged from about `0.395x` to `0.618x` of the CPU
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
