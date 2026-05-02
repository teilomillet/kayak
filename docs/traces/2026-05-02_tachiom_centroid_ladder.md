# Tachiom Centroid-Scale Ladder

Date: `2026-05-02`

## Claim Under Test

Increasing centroid count should move Kayak's bounded streaming Tachiom path
closer to the paper's scale regime.

Reason:
- the paper uses millions of centroids, while the previous strongest local
  streaming rows used `32768`
- changing only centroid count isolates one reproduction gap before changing
  corpus size, query count, or hardware
- vector count remains explicit because memory, postings, graph size, and query
  quality all depend on it

## Implementation

Primary files:
- `python/scripts/run_tachiom_streaming_centroid_ladder.py`
- `python/kayak_bridge/tachiom_streaming_benchmark.py`
- `python/scripts/sweep_tachiom_streaming_pruning.py`
- `python/scripts/sweep_tachiom_streaming_query_policy.py`

The ladder runner takes an existing encoded snapshot and builds one streaming
TAC/PQ index per centroid count. It can build the HNSW sidecar, benchmark one
or more engines, and reuse one exact MaxSim reference across rows.

This avoids re-encoding the same documents and queries for every centroid
setting. It also writes the JSON summary after every row, so a failed large
centroid point is retained as evidence.

## Validation

Focused commands:

```bash
pixi run python -m py_compile \
  python/kayak_bridge/tachiom_streaming_benchmark.py \
  python/scripts/sweep_tachiom_streaming_pruning.py \
  python/scripts/run_tachiom_streaming_centroid_ladder.py
```

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_streaming_index.py
```

Result: `3` tests passed.

Smoke command:

```bash
pixi run python python/scripts/run_tachiom_streaming_centroid_ladder.py \
  --snapshot .cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/snapshot \
  --output-root .cache/kayak/tachiom_centroid_ladder_smoke \
  --centroid-counts 512 \
  --engines streaming_tac_hnsw_pq_mojo \
  --overwrite \
  --query-limit 8 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --run-exact \
  --max-exact-vector-count 200000 \
  --emit-quiet-mean
```

Smoke result:
- `512` centroids completed
- index build: `3.909744712000247s`
- HNSW graph build: `0.5416743700006919s`
- exact reference source in benchmark row: `provided`

## Docs10000 Ladder

Command shape:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --quiet-checks 2 \
  --sleep-seconds 1 --timeout-seconds 120 --force -- \
  pixi run python python/scripts/run_tachiom_streaming_centroid_ladder.py \
    --snapshot .cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/snapshot \
    --output-root .cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder \
    --centroid-counts 65536,131072,262144 \
    --engines streaming_tac_hnsw_pq_mojo \
    --reuse-existing \
    --query-limit 128 \
    --warmup-iterations 1 \
    --measurement-iterations 3 \
    --run-exact \
    --max-exact-vector-count 1000000 \
    --emit-quiet-mean
```

Artifact:
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/centroid_ladder_summary.json`

Fixed shape:
- documents: `10135`
- document vectors: `739372`
- queries: `128`
- query vectors: `4096`
- engine: `streaming_tac_hnsw_pq_mojo`
- `centroids_per_query_vector`: `120`
- `candidate_k`: `1000`
- stored `candidate_pruning_alpha`: `0.35`
- HNSW defaults used here: `M=16`, `ef_construction=64`, `ef_search=64`

| Centroids | Index build s | HNSW build s | Index bytes incl graph | Mean candidate window | QPS | MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `32768` baseline | existing artifact | existing artifact | previous row | `193.171875` | `69.01278987944805` | `0.9895833333333334` | `0.9484375000000002` | `0.8867187500000006` |
| `65536` | `92.35611546399741` | `76.08570662199782` | `74204399` | `163.171875` | `69.72040770584155` | `0.9854910714285714` | `0.8960937500000004` | `0.8414062500000004` |
| `131072` | `188.8298111639997` | `142.48099772999922` | `116470995` | `128.7734375` | `77.63396980050325` | `0.9817708333333334` | `0.8664062499999999` | `0.828125` |
| `262144` | `414.64413611200143` | `291.1031314319989` | `201028083` | `89.6953125` | `89.7766500938627` | `0.9856770833333334` | `0.8460937499999999` | `0.8109374999999999` |

## Interpretation

Validated:
- `262144` centroids is physically materializable on this machine for the
  docs10000 bounded snapshot
- build time and graph time grow quickly but did not fail at `262144`
- larger centroid counts shrink the candidate window and improve bounded query
  throughput under the fixed policy

Not validated:
- larger centroid count did not improve exact top-10 overlap under fixed
  `centroids_per_query_vector=120` and `ef_search=64`
- these rows still do not reproduce the paper-scale centroid counts of roughly
  `2M` to `4M`

Decision:
- do not claim centroid scaling alone improves quality
- the follow-up query-policy co-sweep did recover the `32768` baseline
  final-recall floor on larger centroid artifacts
- that recovery required larger query-side budgets and did not beat the
  `32768` baseline QPS, so larger centroid count is not yet a quality-preserving
  speedup on this bounded docs10000 slice

Follow-up:
- [2026-05-02 Tachiom query-policy co-sweep](2026-05-02_tachiom_query_policy_sweep.md)
