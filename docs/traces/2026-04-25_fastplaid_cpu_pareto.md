# 2026-04-25: FastPlaid CPU Pareto Sweep

## Claim

Before moving to GPU, Kayak needs CPU evidence across more than one synthetic
shape and more than one approximation budget.

The decision rule for this trace is Pareto dominance per shape:

- maximize recall@10 against Kayak exact
- maximize query QPS
- minimize index bytes

Reason:

- speed alone can hide recall loss
- recall alone can hide an unusable latency or memory cost
- comparing one shape can overfit the implementation to the benchmark

## Implementation

Added:

- `python/scripts/bench_fastplaid_cpu_pareto.py`
- `python/tests/test_fastplaid_cpu_pareto.py`

Changed:

- `pyproject.toml`
- `docs/search_layer_optimization_scorecard.md`

The harness:

- runs Kayak exact once per shape as the reference
- runs several Kayak Mojo PLAID-style budgets per shape
- runs FastPlaid once per shape when requested
- records explicit document/query vector counts
- emits per-shape Pareto fronts over recall, QPS, and index bytes
- records whether any Kayak approximation point dominates FastPlaid

## Validation

Focused tests:

```bash
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_fastplaid_cpu_pareto -v
```

Observed:

- `5/5` tests passed

Scorecard run:

```bash
pixi run bench_fastplaid_cpu_pareto
```

Observed:

- artifact: `.cache/kayak/fastplaid_cpu_pareto_scorecard/summary.json`
- quiet-wrapper log: `.cache/kayak/bench_quiet/20260425T104938Z`
- wrapper parsed `Mean: 0.0517302500011283`

## Measured Results

Controls:

- shape set: `scorecard`
- Kayak config set: `scorecard`
- FastPlaid version: `1.4.6.2110`
- FastPlaid device: `cpu`
- FastPlaid nbits: `4`
- warmup iterations: `0`
- measurement iterations: `1`
- seed: `7`

Per-shape FastPlaid dominance check:

| Shape | Total doc vectors | Total query vectors | FastPlaid recall / qps / bytes | Dominating Kayak config | Kayak recall / qps / bytes | Kayak qps vs FastPlaid |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `small_128d_16dv_4q_8qv` | `2048` | `32` | `0.7249999999999999` / `245.97031442805334` / `1499796` | `recall` | `1.0` / `3315.603154127818` / `1066968` | `13.479688237328478` |
| `medium_256d_32dv_8q_16qv` | `8192` | `128` | `0.6375` / `99.05178469180652` / `5716542` | `recall` | `0.7500000000000001` / `942.5719203689406` / `4255712` | `9.515950906908893` |
| `token_128d_300dv_4q_50qv` | `38400` | `200` | `0.575` / `56.37008369349361` / `26006643` | `recall` | `1.0` / `57.985721015479534` / `19777248` | `1.028661254625251` |

Per-shape Pareto fronts:

- `small_128d_16dv_4q_8qv`: Kayak `recall`, `balanced`, and `light`
- `medium_256d_32dv_8q_16qv`: Kayak `recall`, `balanced`, and `light`
- `token_128d_300dv_4q_50qv`: Kayak `recall`, `balanced`, and `light`

FastPlaid did not appear on the per-shape Pareto front in this run.

## Interpretation

- On this CPU scorecard, Kayak has a Pareto point that dominates FastPlaid for
  every measured shape.
- The long-token shape is the tightest result: Kayak `recall` is only `1.029x`
  FastPlaid QPS, although it has higher recall and lower bytes.
- The current CPU evidence favors Kayak, but the long-token shape needs more
  repetitions and more candidate budgets before we should treat this as stable.

## Next CPU Work

Before GPU work, repeat the long-token shape with:

- `measurement_iterations >= 3`
- candidate budgets around `96`, `128`, `160`, and `192`
- normalized vectors enabled and disabled
- at least one larger document-count shape if local runtime remains acceptable

## Long-Token Candidate Sweep

Command:

```bash
pixi run bench_fastplaid_cpu_long_token
```

Observed:

- artifact: `.cache/kayak/fastplaid_cpu_long_token/summary.json`
- quiet-wrapper log: `.cache/kayak/bench_quiet/20260425T115910Z`
- quiet wait timed out and the command was forced under host load
- wrapper parsed `Mean: 0.802741263666879`

Controls:

- shape set: `long_token`
- Kayak config set: `long_token`
- FastPlaid version: `1.4.6.2110`
- FastPlaid device: `cpu`
- FastPlaid nbits: `4`
- warmup iterations: `1`
- measurement iterations: `3`
- seed: `7`

Measured shape:

- documents: `128`
- document vectors/document: `300`
- total document vectors: `38400`
- queries: `4`
- query vectors/query: `50`
- total query vectors: `200`
- vector dim: `128`
- top_k: `10`

Results:

| System/config | Candidate k | Query batch mean s | Query q/s | Index bytes | Recall@10 vs Kayak exact | QPS vs FastPlaid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kayak `candidate_96` | `96` | `0.055075652666952614` | `72.62737355448799` | `19777248` | `0.675` | `1.2860867897866648` |
| Kayak `candidate_128` | `128` | `0.06960272233239569` | `57.46901652635865` | `19777248` | `1.0` | `1.0176623407857508` |
| Kayak `candidate_160` | `160` | `0.06955270866577241` | `57.5103411029115` | `19777248` | `1.0` | `1.0183941171036466` |
| Kayak `candidate_192` | `192` | `0.06865026433297317` | `58.266345204424425` | `19777248` | `1.0` | `1.0317814508374057` |
| FastPlaid CPU | n/a | `0.07083206933384645` | `56.47159595390554` | `26006107` | `0.6` | `1.0` |

Interpretation:

- `candidate_96` dominates FastPlaid on recall, QPS, and index bytes for this
  repeated long-token CPU sweep
- `candidate_128`, `candidate_160`, and `candidate_192` all reach `1.0`
  recall against Kayak exact on this synthetic shape
- the fastest `1.0`-recall Kayak candidate is faster than FastPlaid on CPU
  QPS (`1.032x`) while also using fewer bytes
- all full-window rows use the same fast path because `candidate_k` is greater
  than or equal to the document count, so the minor ordering among
  `candidate_128`, `candidate_160`, and `candidate_192` is measurement noise
  rather than a meaningful candidate-window effect

Longer confirmation:

- artifact: `.cache/kayak/fastplaid_cpu_long_token_confirm/summary.json`
- quiet-wrapper log: `.cache/kayak/bench_quiet/20260425T115937Z`
- quiet wait also timed out under high host load
- controls: `2` warmups, `7` measurement iterations
- `candidate_128`: `0.06847439857145739` batch seconds,
  `58.41599318065928` QPS, `1.0` recall, `19777248` index bytes
- FastPlaid CPU: `0.06994588699931878` batch seconds,
  `57.187065195684156` QPS, `0.5750000000000001` recall,
  `26004937` index bytes
- confirmation ratio: Kayak `candidate_128` is
  `1.0214896144918426x` FastPlaid QPS at exact-reference recall

Repeated paired confirmation:

- artifacts: `.cache/kayak/fastplaid_cpu_long_token_repeated/run-1.json`,
  `.cache/kayak/fastplaid_cpu_long_token_repeated/run-2.json`,
  `.cache/kayak/fastplaid_cpu_long_token_repeated/run-3.json`
- controls: `2` warmups, `9` measurement iterations per run

| Run | Kayak `candidate_96` QPS / recall | Kayak `candidate_128` QPS / recall | FastPlaid QPS / recall | `candidate_128` QPS vs FastPlaid | FastPlaid dominated |
| ---: | ---: | ---: | ---: | ---: | --- |
| `1` | `70.97426795700716` / `0.675` | `56.954691949042115` / `1.0` | `51.58815696043152` / `0.55` | `1.1040264918308047` | `true` |
| `2` | `72.44414913861135` / `0.675` | `58.08360015219432` / `1.0` | `52.25004795161204` / `0.55` | `1.1116468296064463` | `true` |
| `3` | `72.16983959386891` / `0.675` | `57.70317725098399` / `1.0` | `57.629314049721394` / `0.525` | `1.0012816949582095` | `true` |

Interpretation:

- the exact-recall CPU point stayed ahead of FastPlaid in every repeated paired
  run
- run `3` is close enough that the honest public claim should be "Kayak is
  ahead on this measured CPU Pareto shape" rather than a broad speed claim
  across all CPU conditions
