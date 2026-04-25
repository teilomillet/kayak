# 2026-04-25: Mojo i8 PLAID CPU Matrix Gate

## Claim

Kayak can compete with FastPlaid on CPU speed and bytes by adding an explicit
compressed PLAID-style score-proxy payload, while keeping exact search as the
default.

The decision rule is per explicit vector-count shape:

- maximize recall@10 against Kayak exact
- maximize query QPS
- minimize index bytes

Reason:

- an approximation win is not meaningful without exact-reference recall
- a speed win is not enough if the index bytes are worse
- the approximation must be opt-in because i8 rerank changes scoring semantics

## Implementation

Added:

- `kayak/search/plaid_i8_approx_dim128.mojo`

Changed:

- `kayak/search/__init__.mojo`
- `python/kayak_bridge/_mojo_exact_cpu_bindings.mojo`
- `python/kayak_bridge/plaid_approx.py`
- `python/scripts/bench_fastplaid_cpu_pareto.py`
- `python/scripts/bench_fastplaid_speed_track.py`
- `python/tests/test_plaid_approx_public_api.py`
- `python/tests/test_fastplaid_cpu_pareto.py`
- `tests/test_hybrid_flat_dim128.mojo`
- `python/kayak/README.md`
- `docs/product_direction.md`
- `docs/search_layer_optimization_scorecard.md`

The i8 payload stores:

- int8 document token codes
- one `ScoreScalar` scale per document token
- sampled-centroid posting metadata
- document ids and document offsets

It intentionally does not store exact float token values in the prepared i8
index after build. Reason: the bytes claim must reflect the compressed score
proxy rather than an exact rerank payload hidden behind the same API.

The rerank kernel scores four consecutive document tokens per query-vector
loop. Reason: this reuses each loaded query-vector SIMD block across four token
dots and reduces repeated query-vector loads in the high document-vector-count
shapes.

## Validation

Correctness and API checks:

```bash
pixi run test_hybrid_flat_dim128
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_fastplaid_cpu_pareto -v
```

Observed:

- Mojo hybrid-flat tests: `10/10` passed
- FastPlaid CPU Pareto config tests: `8/8` passed

Benchmark checks:

```bash
pixi run bench_fastplaid_cpu_matrix_v2_smoke
pixi run bench_fastplaid_cpu_matrix_v2
```

Artifacts:

- `.cache/kayak/fastplaid_cpu_matrix_v2_smoke/summary.json`
- `.cache/kayak/fastplaid_cpu_matrix_v2/summary.json`
- full matrix quiet log: `.cache/kayak/bench_quiet/20260425T133749Z`

Host-load caveat:

- the full matrix quiet wait timed out under competing host CPU load and then
  ran with `--force`
- this is still a completed full matrix artifact, but tight rows should be
  repeated under quieter load before using them as release marketing numbers

## Measured Results

Full `cpu_matrix_v2` result:

- `18/18` raw and normalized shape rows have a Kayak i8 point that dominates
  FastPlaid on recall@10, query QPS, and index bytes.
- covered document counts: `128`, `512`, `2048`
- covered document vectors/document: `32`, `128`, `300`
- covered query vectors/query: `16`, `50`, `96`
- vector dim: `128`
- FastPlaid: CPU, `nbits=4`, `1.4.6.2110`

Tight rows:

| Shape | FastPlaid recall / qps / bytes | Kayak i8 config | Kayak recall / qps / bytes | QPS vs FastPlaid |
| --- | ---: | --- | ---: | ---: |
| `small_128d_300dv_4q_96qv_raw` | `0.625` / `49.557` / `26005345` | `i8_ratio_75pct` | `0.825` / `49.774` / `5185248` | `1.004x` |
| `large_2048d_128dv_4q_50qv_normalized` | `0.475` / `13.236` / `41970080` | `i8_ratio_725pct` | `0.75` / `14.217` / `35947696` | `1.074x` |
| `small_128d_300dv_4q_96qv_normalized` | `0.775` / `46.179` / `26017427` | `i8_ratio_725pct` | `0.825` / `50.284` / `5190216` | `1.089x` |

## Interpretation

Verified:

- the previous CPU matrix gap is closed in the completed synthetic CPU matrix
- the compressed i8 payload is the reason the bytes axis moved in Kayak's
  favor
- the four-token Mojo rerank loop materially improved the token-heavy rows

Not claimed:

- this is not a GPU result
- this is not a real-corpus quality claim
- this does not make approximation the default path

Next gates:

- repeat the tight CPU rows under lower host load
- add the GPU backend behind the same explicit `PlaidApproxConfig` contract
- keep exact MaxSim as the correctness reference for every approximate lane
