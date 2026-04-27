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
- after: `.cache/kayak/bench_quiet/20260427T145047Z`
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
| `query_vectors32` | `0.23629640221274195` | `0.2342169343180451` | `0.41620342988766384` | `0.4157538015884203` |
| `doc_vectors64` | `0.31803729470704345` | `0.3084105431618968` | `0.6313771376696127` | `0.6252696262295903` |
| `query_batch4` | `0.24211193416623314` | `0.2349893458755944` | `0.571942692878157` | `0.5557679641520378` |

After-change timings:

| case | score extension s | score host+extension s | prepare extension s |
| --- | ---: | ---: | ---: |
| `query_vectors32` | `0.00009333466732641682` | `0.00009446666808798909` | `0.000547851999726845` |
| `doc_vectors64` | `0.00011298733321988645` | `0.00011410633427052139` | `0.0018424569971102756` |
| `query_batch4` | `0.0001347946672467515` | `0.00013597333357514194` | `0.0005445160022645723` |

## Interpretation

The change is accepted, but only as a small local improvement. It preserved
correctness and reduced the worst non-full score-call ratio from about `0.318x`
to `0.308x` of CPU candidate generation. It did not change the architecture
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
