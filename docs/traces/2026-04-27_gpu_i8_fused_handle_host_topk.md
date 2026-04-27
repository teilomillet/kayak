# 2026-04-27: GPU I8 Fused Handle Host Top-K Scan

## Question

Does the prepared fused handle spend enough time in host top-k for a scoped
serving-path top-k change to improve the score call?

## Change

Changed the prepared fused handle score path from non-destructive host top-k
with duplicate checks to destructive host top-k after document-score readback.

Reason: each prepared-handle score call overwrites `document_scores_host` from
the device before top-k runs. That makes it safe for the serving-shaped
no-reference top-k path to mark selected scores as negative infinity and avoid
checking the previously selected ranks for every document. The non-destructive
helper remains available for profiling contexts that need to repeat top-k over
the same score buffer.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_fused_centroid_posting_handle
```

Artifacts:

- before: `.cache/kayak/bench_quiet/20260427T144646Z`
- after: `.cache/kayak/bench_quiet/20260427T150112Z`
- report: `.cache/kayak/gpu_i8_fused_centroid_posting_handle/summary.json`

Status after the change:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- minimum top-k position agreement: `1.0`
- maximum top-k position mismatch: `0`
- maximum top-k score delta: `0.00006103515625`

Non-full score-call ratios:

| case | before score / CPU candidate | after score / CPU candidate | before score / CPU slice | after score / CPU slice |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.23629640221274195` | `0.22351480984486652` | `0.41620342988766384` | `0.3954280157885903` |
| `doc_vectors64` | `0.31803729470704345` | `0.30692159557252907` | `0.6313771376696127` | `0.6176179939835656` |
| `query_batch4` | `0.24211193416623314` | `0.23600695975367733` | `0.571942692878157` | `0.5483388835915574` |

After-change timings:

| case | score extension s | score host+extension s | prepare extension s |
| --- | ---: | ---: | ---: |
| `query_vectors32` | `0.00009281333283676456` | `0.00009400866595872989` | `0.0005263919993012678` |
| `doc_vectors64` | `0.00011206566826634419` | `0.00011310433546896093` | `0.0018464940003468655` |
| `query_batch4` | `0.00013698866678168997` | `0.00013820433377986774` | `0.0005343660013750196` |

## Interpretation

The change is accepted, but only as a small local improvement. It preserved
correctness and reduced the worst non-full score-call ratio from about `0.318x`
to `0.307x` of CPU candidate generation. It did not change the architecture
conclusion: the prepared fused handle is already faster than the CPU target on
the non-full rows, and further optimization needs a prepared-handle substep
profile rather than more guessing.

## Next Step

Add a prepared-handle score breakdown that measures query H2D, centroid scoring,
device heap selection, accumulation, reduction, document-score readback, and
host top-k under the same explicit handle.

Reason: the accepted top-k change was selected from prior profile evidence, but
the next larger improvement should be chosen from timings collected at the
exact prepared-handle boundary.
