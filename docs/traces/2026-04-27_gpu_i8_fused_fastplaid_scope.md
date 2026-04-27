# 2026-04-27: GPU I8 Fused FastPlaid Scope Check

## Question

Can the fused centroid-posting GPU primitive with block-parallel device top-k
be treated as the preferred FastPlaid-facing search path?

## Change

Added a separate FastPlaid-scope report row for the fused GPU i8
centroid-posting primitive.

Reason: the older FastPlaid policy matrix measured CPU candidate generation
plus the address-window GPU top-k primitive. The fused primitive has a different
boundary: it performs centroid scoring, selected-posting accumulation, document
score reduction, and top-k inside the prepared GPU payload without accepting
CPU candidate positions. Measuring it as a separate row avoids mixing two
different algorithms.

## Measurement

Command:

```bash
pixi run compare_gpu_i8_fastplaid_policy
```

Artifacts:

- report: `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T152615Z`

Status:

- benchmark status: `ok`
- rows: `6 / 6`
- fused device top-k position agreement min versus CPU fused reference: `1.0`
- address-window top-k position agreement min: `1.0`

Rows:

| case | FastPlaid device | address envelope / FastPlaid | address recall | fused device top-k / FastPlaid | fused recall | FastPlaid recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `cpu` | `0.03882287244855747` | `0.7` | `0.0053551645900365` | `0.1` | `0.35` |
| `query_vectors32` | `cuda` | `0.21635962087098493` | `0.7` | `0.030143771197010175` | `0.1` | `0.4` |
| `doc_vectors64` | `cpu` | `0.025612052394439234` | `0.65` | `0.004122929583067248` | `0.0` | `0.5` |
| `doc_vectors64` | `cuda` | `0.10724984396028117` | `0.65` | `0.017196016528468566` | `0.0` | `0.6499999999999999` |
| `query_batch4` | `cpu` | `0.048620027845971095` | `0.7` | `0.006714434660536522` | `0.05` | `0.625` |
| `query_batch4` | `cuda` | `0.1362696441531551` | `0.7` | `0.01870384240286398` | `0.05` | `0.6` |

## Interpretation

The fused device-top-k path is fast but not recall-competitive as a final
search path. It measured between about `0.0041x` and `0.0301x` of FastPlaid
full-search time on the six rows, but recall was only `0.0` to `0.1` against
Kayak exact while FastPlaid measured `0.35` to `0.65` and the address-window
Kayak envelope measured `0.65` to `0.7`.

This accepts the fused device top-k implementation as an exact implementation
of the current fused approximate score, but rejects promoting that approximate
score directly to final search output.

## Next Step

Use fused centroid-posting scores as a GPU candidate/pruning stage, then run an
exact candidate rerank before final top-k.

Reason: the measured failure is quality, not GPU mechanics. The address-window
path keeps recall because it reranks retained documents with the full i8
candidate scorer. The next primitive should therefore test whether a fused GPU
shortlist plus exact rerank can keep the address-window recall while removing
the CPU candidate-generation cost.
