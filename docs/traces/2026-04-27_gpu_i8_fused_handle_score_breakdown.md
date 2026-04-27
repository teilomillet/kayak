# 2026-04-27: GPU I8 Fused Handle Score Breakdown

## Question

After the fused primitive moved behind an explicit prepared handle, which
substep dominates the serving-shaped score call?

## Change

Added a prepared-handle score profile path that measures:

- query host ingest
- query H2D
- centroid scoring kernel
- device heap centroid-selection kernel
- selected-posting accumulation kernel
- document-score reduction kernel
- document-score D2H
- destructive host top-k estimate
- non-destructive host top-k comparison

Reason: the prepared handle already wins on the non-full rows. The next
optimization should target the largest measured remaining cost at the actual
prepared-handle boundary instead of relying on the earlier in-call resident
probe.

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
- minimum top-k position agreement: `1.0`
- maximum top-k position mismatch: `0`
- maximum top-k score delta: `0.00006103515625`

Non-full score-call results:

| case | cpqv | score extension s | score / CPU candidate | score / CPU slice |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `4` | `0.00009281333283676456` | `0.22351480984486652` | `0.3954280157885903` |
| `doc_vectors64` | `16` | `0.00011206566826634419` | `0.30692159557252907` | `0.6176179939835656` |
| `query_batch4` | `8` | `0.00013698866678168997` | `0.23600695975367733` | `0.5483388835915574` |

Non-full prepared-handle profile:

| case | query H2D s | centroid score s | centroid select s | accumulation s | reduction s | D2H s | destructive top-k est. s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.00000520183134663557` | `0.000016861269208037827` | `0.000014713140441176471` | `0.000007697653256336221` | `0.000004987735366208301` | `0.000003503366666666667` | `0.00003181604368355359` |
| `doc_vectors64` | `0.000004266919096657633` | `0.000011327801988636364` | `0.000027846246873552573` | `0.000020393855731561914` | `0.000004251107366632447` | `0.0000035315001333333332` | `0.00003181445289369919` |
| `query_batch4` | `0.000006841301661356395` | `0.000010978032319391635` | `0.000019058231893265565` | `0.00001075446132497762` | `0.00000425712538962879` | `0.0000036394371666666668` | `0.00006365121981363766` |

## Interpretation

Transfers are not the limiting cost on these rows. Query H2D plus document-score
D2H is about `8.7us`, `7.8us`, and `10.5us` on the non-full rows.

Host top-k is the largest measured substep on `query_vectors32` and
`query_batch4`, and tied with the GPU selection/accumulation area on
`doc_vectors64`. The previous one-lane device top-k remains rejected, so the
next top-k attempt should be a block-parallel device top-k or another measured
parallel selection scheme, not a serial GPU scan.

The profile method intentionally synchronizes individual kernels to measure
substeps. The serving score path keeps the kernels fused behind one
synchronization after the kernel chain.

## Next Step

Prototype a block-parallel device document top-k for the prepared fused handle
and keep the host destructive top-k as the reference.

Reason: the evidence now points at final document top-k as the most repeated
remaining cost, but the earlier serial device top-k failure shows that only a
parallel top-k design is worth testing.
