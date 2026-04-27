# 2026-04-27: GPU I8 Fused Block-Parallel Device Top-K

## Question

Can a block-parallel device document top-k beat the prepared fused handle's
host destructive top-k path?

## Change

Added an alternate benchmark-only prepared-handle score path that keeps dense
document scores on device and runs a block-parallel top-k kernel before reading
back only `[query_count, top_k]` positions and scores.

Reason: the previous one-lane device top-k was rejected because it serialized
the document scan on one GPU lane per query. The prepared-handle score breakdown
showed host top-k was now a leading cost, so the next valid test was a parallel
device selection design, not another serial scan.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_fused_centroid_posting_handle
```

Artifacts:

- report: `.cache/kayak/gpu_i8_fused_centroid_posting_handle/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T151550Z`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- host top-k mismatch max: `0`
- device top-k mismatch max: `0`
- device top-k score delta max: `0.00006103515625`

Non-full rows:

| case | host score s | device-top-k score s | device / host | device / CPU candidate |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.00009607533381010096` | `0.00008196633522553991` | `0.8531465046746713` | `0.19982593862816195` |
| `doc_vectors64` | `0.00011333500030256498` | `0.00009867099894715163` | `0.8706136558321298` | `0.2765292962040435` |
| `query_batch4` | `0.00013530533275722215` | `0.0000889523325895425` | `0.6574192663133931` | `0.1541108712367938` |

## Interpretation

The block-parallel device top-k is accepted as a measured internal alternate.
It preserved exact top-k order on the wide matrix and improved the score path
on every non-full row. The biggest gain is the `query_batch4` row, where
avoiding host top-k and full document-score readback reduces the score call to
about `0.657x` of the host-top-k path.

This does not make the old one-lane device top-k valid. That rejected design
was serial. The accepted design uses one block per query and parallel per-block
selection over document scores.

## Next Step

Promote the device-top-k path to the preferred internal prepared fused score
path only after one more validation pass compares it against FastPlaid scope
rows and records the final score matrix.

Reason: the primitive now has a measured win over the host-top-k score path, but
the repo should keep the public boundary unchanged until the full comparison
matrix is updated with the new preferred internal path.
