# 2026-04-25: CPU Matrix v2 Smoke And GPU Gate

## Claim

Kayak should not move the main implementation focus to GPU until the CPU
comparison against FastPlaid is broadened enough to show where Kayak is already
Pareto-favorable and where it is not.

The decision rule for this trace is per-shape Pareto dominance:

- maximize recall@10 against Kayak exact
- maximize query QPS
- minimize index bytes

Reason:

- speed without exact-reference recall can hide a weak candidate stage
- recall without bytes can hide an uncompetitive storage profile
- GPU work should accelerate an understood plan, not hide an unresolved CPU
  tradeoff

## Implementation

Changed:

- `python/scripts/bench_fastplaid_cpu_pareto.py`
- `python/tests/test_fastplaid_cpu_pareto.py`
- `python/kayak_bridge/plaid_approx.py`
- `python/kayak_bridge/late_ops.py`
- `python/kayak_bridge/late_index.py`
- `python/kayak_bridge/late_query_batch.py`
- `python/kayak/README.md`
- `docs/search_layer_optimization_scorecard.md`
- `docs/product_direction.md`

Added:

- `python/tests/test_plaid_approx_public_api.py`

The harness now has:

- `cpu_matrix_v2_smoke`: three explicit vector-count shapes
- `cpu_matrix_v2`: nine explicit vector-count shapes
- raw and normalized sweeps through `--normalization-set both`
- fixed, ratio-based, and full-window Kayak candidate budgets
- report fields for query-QPS ratios against exact and FastPlaid

The public API now has:

- `PlaidApproxConfig`
- `PlaidApproxIndex`
- `prepare_plaid_approx_index(...)`
- `search(..., approximation=...)`
- `search_batch(..., approximation=...)`

Reason:

- approximation is a user-facing speed, recall, and bytes tradeoff
- exposing it as a parameter keeps exact MaxSim as the default and prevents
  hidden semantic changes
- prepared approximation reuse lets callers separate build cost from repeated
  query cost

## Validation

Focused checks:

```bash
pixi run test_hybrid_flat_dim128
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_plaid_approx_public_api \
  python.tests.test_fastplaid_cpu_pareto \
  python.tests.test_fastplaid_speed_track \
  python.tests.test_public_api_contract -v
```

Observed:

- `9/9` Mojo hybrid-flat tests passed
- `20/20` focused Python tests passed

Benchmark:

```bash
pixi run bench_fastplaid_cpu_matrix_v2_smoke
```

Observed:

- artifact: `.cache/kayak/fastplaid_cpu_matrix_v2_smoke/summary.json`
- quiet-wrapper log: `.cache/kayak/bench_quiet/20260425T124919Z`
- the quiet wait timed out under host load and the benchmark was forced
- controls: `1` warmup, `2` measurement iterations, seed `7`

## Measured Results

| Shape | Total doc vectors | Total query vectors | FastPlaid recall / qps / bytes | Kayak full-window recall / qps / bytes | Kayak `ratio_25pct` recall / qps / bytes | FastPlaid dominated |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `small_128d_32dv_4q_16qv_raw` | `4096` | `64` | `0.675` / `177.541217912271` / `2992421` | `1.0` / `1506.8986764240287` / `2128784` | `0.275` / `2411.273428993025` / `2128784` | `true` |
| `medium_512d_128dv_4q_50qv_raw` | `65536` | `200` | `0.4` / `22.7221247459711` / `44594563` | `1.0` / `34.10114796060502` / `33880728` | `0.225` / `114.73303543968606` / `33880728` | `true` |
| `large_2048d_32dv_4q_96qv_raw` | `65536` | `384` | `0.325` / `16.837873639591965` / `11031386` | `1.0` / `17.650832324853674` / `34032104` | `0.25` / `49.44983198773549` / `34032104` | `false` |
| `small_128d_32dv_4q_16qv_normalized` | `4096` | `64` | `0.7` / `169.68970646510937` / `2993053` | `1.0` / `1548.199715242828` / `2129104` | `0.375` / `2444.30199443263` / `2129104` | `true` |
| `medium_512d_128dv_4q_50qv_normalized` | `65536` | `200` | `0.625` / `23.460152633396607` / `44600036` | `1.0` / `33.85741193053026` / `33892312` | `0.30000000000000004` / `112.37809609863578` / `33892312` | `true` |
| `large_2048d_32dv_4q_96qv_normalized` | `65536` | `384` | `0.7` / `18.121671776568448` / `11032755` | `1.0` / `17.638699921365276` / `34038016` | `0.30000000000000004` / `49.04218347541088` / `34038016` | `false` |

## Interpretation

Verified:

- Kayak is Pareto-favorable against FastPlaid on the small and medium v2 smoke
  shapes in both raw and normalized modes.
- Kayak is not Pareto-favorable on the large v2 smoke shape. FastPlaid remains
  on the front because it uses far fewer bytes and has better recall than
  Kayak's pruned rows. Kayak's full-window row has exact recall, but its stored
  exact-vector payload keeps it from dominating on bytes.

Debunked:

- The current evidence does not support a broad claim that CPU coverage is
  already in Kayak's favor across the expanded comparison.

Next CPU gates:

- improve pruned candidate recall on high-document-count, high-query-vector
  shapes
- add a compressed payload or compressed score-proxy lane and report its bytes
- rerun the full `cpu_matrix_v2` suite before making GPU speed claims
