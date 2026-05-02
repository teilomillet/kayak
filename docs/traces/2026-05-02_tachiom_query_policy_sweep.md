# Tachiom Query-Policy Co-Sweep

Date: `2026-05-02`

## Claim Under Test

The centroid-scale ladder showed that larger centroid artifacts can reduce the
candidate window and improve bounded throughput under a fixed query policy, but
exact top-10 overlap falls. This sweep tests whether query-side policy budgets
can recover the `32768`-centroid quality floor on the `131072` and `262144`
centroid artifacts.

Reason:
- the paper reports a grid over centroid budget, `k_c`, `k_d`, and candidate
  pruning `alpha`
- the fixed-policy ladder changed centroid count but held
  `centroids_per_query_vector=120`, `ef_search=64`, and stored pruning policy
- changing query-side policy avoids rebuilding TAC/PQ and HNSW sidecars, so the
  result isolates serving-time budget from build-time centroid scale

## Implementation

Primary files:
- `python/kayak_bridge/tachiom_streaming_benchmark.py`
- `python/scripts/bench_tachiom_streaming_index.py`
- `python/scripts/sweep_tachiom_streaming_query_policy.py`
- `python/tests/test_tachiom_streaming_index.py`

The benchmark now supports query-time overrides for:
- `centroids_per_query_vector`
- HNSW `ef_search`
- `candidate_pruning_alpha`

The overrides replace only lightweight frozen metadata wrappers. Prepared native
Mojo handles, graph arrays, and memmapped payloads are preserved. That keeps the
sweep measuring query policy instead of reload or rebuild cost.

## Validation

Focused commands:

```bash
pixi run python -m py_compile \
  python/kayak_bridge/tachiom_streaming_benchmark.py \
  python/scripts/bench_tachiom_streaming_index.py \
  python/scripts/sweep_tachiom_streaming_pruning.py \
  python/scripts/sweep_tachiom_streaming_query_policy.py \
  python/tests/test_tachiom_streaming_index.py
```

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_streaming_index.py
```

Result: `3` tests passed.

Smoke command:

```bash
pixi run python python/scripts/sweep_tachiom_streaming_query_policy.py \
  --snapshot .cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/snapshot \
  --index .cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/streaming_tachiom_index_c131072 \
  --engine streaming_tac_hnsw_pq_mojo \
  --query-limit 8 \
  --centroids-per-query-vector-values 120,180 \
  --hnsw-ef-search-values 64 \
  --candidate-pruning-alphas 0.35 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --run-exact \
  --max-exact-vector-count 1000000 \
  --output .cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_smoke_c131072.json \
  --emit-quiet-mean
```

Smoke result:
- fastest smoke row: `kc120_ef64_alpha_0p35`
- fastest smoke row batch mean: `0.10962405600002967s`
- best smoke final recall@10 vs exact: `0.8125000000000001`
- exact reference source: `provided`
- `centroids_per_query_vector` and `ef_search` sources: `override`

## Docs10000 Co-Sweep

Quality floors:
- primary metric floor: `0.989`
- final recall@10 vs exact floor: `0.8867`

The final-recall floor matches the fixed-policy `32768`-centroid baseline on
this bounded docs10000 snapshot.

Command shape:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --quiet-checks 2 \
  --sleep-seconds 1 --timeout-seconds 120 --force -- \
  pixi run python python/scripts/sweep_tachiom_streaming_query_policy.py \
    --snapshot .cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/snapshot \
    --index <streaming_tachiom_index_c131072_or_c262144> \
    --engine streaming_tac_hnsw_pq_mojo \
    --query-limit 128 \
    --centroids-per-query-vector-values 120,180,240,360 \
    --hnsw-ef-search-values 64,128,256 \
    --candidate-pruning-alphas 0.3,0.35,0.5 \
    --warmup-iterations 1 \
    --measurement-iterations 2 \
    --sweep-repeats 1 \
    --run-exact \
    --max-exact-vector-count 1000000 \
    --min-primary-value 0.989 \
    --min-final-recall-at-k-vs-exact 0.8867 \
    --output <query_policy_sweep.json> \
    --emit-quiet-mean
```

Artifacts:
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_sweep_c131072.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_sweep_c262144.json`
- `.cache/kayak/bench_quiet/20260502T173626Z`
- `.cache/kayak/bench_quiet/20260502T174724Z`

Fixed shape:
- documents: `10135`
- document vectors: `739372`
- queries: `128`
- query vectors: `4096`
- engine: `streaming_tac_hnsw_pq_mojo`
- `candidate_k`: `1000`

| Centroids | Row kind | Setting | `k_c` | `ef_search` | Alpha | Mean candidate window | QPS | MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `131072` | fastest | `kc120_ef64_alpha_0p3` | `120` | `64` | `0.3` | `98.4921875` | `87.69168200632623` | `0.9817708333333334` | `0.8304687499999999` | `0.8023437499999998` |
| `131072` | fastest eligible | `kc180_ef128_alpha_0p5` | `180` | `128` | `0.5` | `296.9375` | `40.093095874216985` | `0.9895833333333334` | `0.9484375000000004` | `0.8890625000000004` |
| `262144` | fastest | `kc120_ef64_alpha_0p3` | `120` | `64` | `0.3` | `70.421875` | `97.06290014898175` | `0.9856770833333334` | `0.8257812499999998` | `0.7945312499999997` |
| `262144` | fastest eligible | `kc360_ef64_alpha_0p35` | `360` | `64` | `0.35` | `136.2421875` | `24.116374186902767` | `0.9934895833333334` | `0.9421875000000004` | `0.8875000000000008` |

Additional observations:
- `131072` centroids had `9` rows meeting the final-recall floor and `32` rows
  meeting the primary metric floor.
- `262144` centroids had `12` rows meeting the final-recall floor and `9` rows
  meeting the primary metric floor.
- Increasing `ef_search` from `64` to `128` or `256` did not materially improve
  the best eligible rows in this grid.
- The dominant quality/speed controls in this grid were `k_c` and pruning
  `alpha`, not `ef_search`.

## Interpretation

Validated:
- quality can be recovered on larger centroid artifacts by increasing query-side
  budget
- the fastest rows on both larger artifacts are not quality preserving under the
  explicit floors
- no measured larger-centroid row in this grid both preserves quality and beats
  the fixed-policy `32768` baseline QPS of `69.01278987944805`

Not validated:
- larger centroid counts are not yet a quality-preserving speedup on the
  bounded docs10000 slice
- this does not reproduce the paper's paper-scale centroid regime or throughput
  claim

Decision:
- keep the query-policy sweep as the next guardrail for larger centroid
  artifacts
- do not claim `131072` or `262144` centroids improve the bounded result unless
  the claim is explicitly scoped to speed-only rows that fail exact-overlap or
  MRR floors
- the next optimization should change scoring/traversal cost or build policy,
  not just widen `k_c` or `ef_search`

Follow-up:
- [2026-05-02 Tachiom HNSW heap frontier](2026-05-02_tachiom_hnsw_heap_frontier.md)
  reduced traversal cost and improved the best measured larger-centroid
  quality-preserving row to `67.76123363711898` QPS. The optimized `32768`
  baseline is still faster at `89.25310082533578` QPS.
