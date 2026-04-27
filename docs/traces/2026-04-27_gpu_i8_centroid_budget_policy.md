# 2026-04-27: GPU I8 Centroid-Budget Policy Replay

## Claim

The selected centroid budget should become a replayed benchmark policy before
it becomes a default.

Reason: the prior sweep found shape-dependent winners, but choosing the best
budget after seeing recall is an oracle. A usable next step must separate
static policies, shape-only policies, and oracle calibration ceilings.

## Change

Added a benchmark-only policy replay layer:

- `python/kayak_bridge/gpu_i8_centroid_budget_policy.py`
- `python/kayak_bridge/gpu_i8_centroid_budget_policy_runner.py`
- `python/scripts/profile_gpu_i8_centroid_budget_policy.py`
- `profile_gpu_i8_centroid_budget_policy_raw`
- `profile_gpu_i8_centroid_budget_policy`

The replay first runs the existing measured centroid-budget sweep, then maps
named policies onto those measured rows. Static and shape-rule policies use
only explicit shape fields: document count, document vectors, query count,
query vectors, and candidate window size. Oracle rows inspect recall after the
sweep and are labelled as calibration ceilings.

Reason: this keeps the optimization falsifiable. If a shape rule loses recall
or only wins on one cherry-picked surface, the report shows that directly.

## Measurement

Commands:

```bash
pixi run python -m py_compile python/kayak_bridge/gpu_i8_centroid_budget_policy.py python/kayak_bridge/gpu_i8_centroid_budget_policy_runner.py python/scripts/profile_gpu_i8_centroid_budget_policy.py
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_centroid_budget_policy.py
pixi run python python/scripts/profile_gpu_i8_centroid_budget_policy.py --case smoke:documents=64,document_vectors=8,queries=1,query_vectors=4,candidate_k=16 --centroid-budgets 4,8,32 --baseline-centroids-per-query-vector 32 --policies static4,static32,shape_rule_v0,oracle_fastest_no_final_recall_loss --measurement-iterations 1 --output .cache/kayak/gpu_i8_centroid_budget_policy/smoke.json
pixi run profile_gpu_i8_centroid_budget_policy
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- pixi run profile_gpu_i8_centroid_budget_policy_raw --case-set default --output .cache/kayak/gpu_i8_centroid_budget_policy/default_summary.json
```

Artifacts:

- smoke report: `.cache/kayak/gpu_i8_centroid_budget_policy/smoke.json`
- wide quiet log: `.cache/kayak/bench_quiet/20260427T100811Z`
- wide report: `.cache/kayak/gpu_i8_centroid_budget_policy/summary_after_split.json`
- default quiet log: `.cache/kayak/bench_quiet/20260427T100906Z`
- default report:
  `.cache/kayak/gpu_i8_centroid_budget_policy/default_summary_after_split.json`

## Results

All wide and default policy replay reports were `ok`.

Wide non-full summary:

| policy | cases | no final recall loss | mean recall | min recall | mean candidate+score / static32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shape_rule_v0` | `3` | `3` | `0.6833333333333332` | `0.65` | `0.8702960769349722` |
| `static32` | `3` | `3` | `0.6499999999999999` | `0.6499999999999999` | `1.0` |
| `static4` | `3` | `1` | `0.6166666666666667` | `0.55` | `0.8483663041403493` |

Default non-full summary:

| policy | cases | no final recall loss | mean recall | min recall | mean candidate+score / static32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shape_rule_v0` | `5` | `5` | `0.43` | `0.15000000000000002` | `0.8650206099220128` |
| `static32` | `5` | `5` | `0.41` | `0.15000000000000002` | `1.0` |
| `static4` | `5` | `3` | `0.39` | `0.15000000000000002` | `0.8117803563092515` |

`shape_rule_v0` selected these budgets:

| case | seed | docs | doc vectors | queries | query vectors | candidate_k | cpqv | recall | candidate+score / static32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `7` | `256` | `16` | `2` | `8` | `128` | `4` | `0.5` | `0.8601034409510804` |
| `candidate32` | `8` | `256` | `16` | `2` | `8` | `32` | `4` | `0.15000000000000002` | `0.7452783538135568` |
| `query_vectors16` | `9` | `256` | `16` | `2` | `16` | `128` | `16` | `0.6499999999999999` | `0.8955769815708813` |
| `doc_vectors32` | `10` | `256` | `32` | `2` | `8` | `128` | `4` | `0.6` | `0.8704235239984537` |
| `documents512` | `11` | `512` | `16` | `2` | `8` | `128` | `24` | `0.25` | `0.9537207492760922` |
| `query_vectors32` | `7` | `512` | `16` | `2` | `32` | `256` | `4` | `0.7` | `0.77471783148268` |
| `doc_vectors64` | `8` | `512` | `64` | `2` | `8` | `256` | `16` | `0.65` | `0.9592826069487204` |
| `query_batch4` | `9` | `512` | `16` | `4` | `8` | `256` | `8` | `0.7` | `0.8768877923735162` |

The tiny smoke shape falsified broader generalization: `shape_rule_v0` chose
`cpqv=4`, ran at `0.8069467160401321x` of static32 candidate+score time, but
lost final recall from `0.4` to `0.3`. The oracle row chose `cpqv=8`.

## Interpretation

Verified:

- a reproducible policy replay report now separates static, shape-only, and
  oracle calibration rows
- on the default and wide non-full matrices, `shape_rule_v0` matched the oracle
  no-final-loss row and preserved static32 final recall on every case
- `shape_rule_v0` reduced mean candidate-generation-plus-score time to about
  `0.87x` of static32 on both matrices
- a single static low budget is not enough: `static4` is faster on average but
  loses final recall on five of the eight default+wide non-full rows

Debunked:

- `shape_rule_v0` is not a general policy over every possible shape; the smoke
  case already shows a recall loss
- the evidence still does not justify changing the public search default

## Decision

Keep `shape_rule_v0` benchmark-only and use it as the next automated FastPlaid
comparison policy.

Reason: it is now stronger than manually selected rows for the measured
default and wide matrices, but it is still a synthetic, same-surface rule. The
next step should automate same-shape FastPlaid CPU/CUDA comparison for the
shape-rule policy instead of hand-running selected rows.

## Validation

Observed:

- Python syntax check passed
- focused policy tests: `5 / 5` passed
- focused GPU/FastPlaid suite including policy tests: `58 / 58` passed
- smoke policy report status: `ok`
- wide policy replay status: `ok`
- default policy replay status: `ok`
