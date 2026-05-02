# Tachiom HNSW Visited Table

Date: `2026-05-03`

## Claim Under Test

The HNSW+PQ query path initializes a sparse visited table for every query
vector. The old `ef * 64 + 16` initial size may over-initialize memory on the
bounded Tachiom rows, but shrinking it too far can add probing and rehash cost.

Reason:
- the larger-centroid rows still spend substantial isolated time in HNSW
  traversal
- the visited table rehashes when needed, so a smaller initial size should not
  change correctness
- the tradeoff is empirical: table initialization, linear probing, and rehashes
  all compete

## Change

The measured HNSW+PQ path now starts the sparse visited table at
`ef * 32 + 16` instead of `ef * 64 + 16`.

This keeps the table large enough for the measured rows while halving the
initial power-of-two allocation in common cases:

| Effective `ef` | Old initial table | New initial table |
| ---: | ---: | ---: |
| `120` | `8192` slots | `4096` slots |
| `200` | `16384` slots | `8192` slots |
| `360` | `32768` slots | `16384` slots |

Correctness reason:
- visited-table growth remains enabled
- selected centroid counts, candidate windows, and rerank policy are unchanged
- the change affects only the initial sparse-set capacity

## Validation

Focused correctness:

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_streaming_index.py
```

Result: `3` tests passed.

The fixture covers normal and address-backed Mojo HNSW+PQ paths against the
Python reference on a dim128 fixture.

## Measurements

All rows below use:
- documents: `10135`
- document vectors: `739372`
- queries: `128`
- query vectors: `4096`
- final `k`: `10`
- exact-reference vector cap: `1000000`
- primary metric floor: `0.989`
- final recall@10 vs exact floor: `0.8867`
- quiet wrapper with `sweep_repeats=3`

| Artifact | Policy | Previous QPS | `ef*32` QPS | MRR@10 | Final recall@10 vs exact |
| --- | --- | ---: | ---: | ---: | ---: |
| `32768` centroids | `kc120_ef64_alpha0.35` | `92.33851822318282` | `92.78163789778101` | `0.9895833333333334` | `0.8867187500000006` |
| `131072` centroids | `kc200_ef64_alpha0.45` | `68.40695596204677` | `68.18820304691724` | `0.9895833333333334` | `0.8906250000000002` |
| `262144` centroids | `kc360_ef64_alpha0.35` | `70.85580779445326` | `73.48173692089176` | `0.9934895833333334` | `0.8875000000000008` |

The `32768` and `262144` previous QPS values are from the controlled rerank
hot-loop A/B. The `131072` previous QPS is a follow-up repeated baseline run
with the same rerank cleanup but the old `ef * 64 + 16` table sizing.

The evidence is mixed: the current fastest `32768` row and the `262144` stress
row improved, while the `131072` eligible row was slightly slower.

Rejected setting:

| Setting | Row | QPS | Decision |
| --- | --- | ---: | --- |
| `ef*16 + 16` | `262144`, `kc360_ef64_alpha0.35` | `69.209911` | rejected; smaller table likely added too much probing or rehash work |

Internal profile, `262144` centroids with `kc360_ef64_alpha0.35`:

| Stage | Batch seconds |
| --- | ---: |
| full search | `1.58750085175558` |
| candidate generation | `0.8806590430277359` |
| HNSW traversal | `0.7824523499387878` |
| residual score table | `0.08142752452136201` |
| rerank scoring | `0.5985668354660718` |

The profile still identifies HNSW traversal as the dominant isolated stage, but
it does not prove that the end-to-end gain comes only from the isolated HNSW
timer. Treat the repeated end-to-end rows as the decision-quality evidence.

Current profile split:

| Artifact | Full search batch s | HNSW traversal share | Rerank scoring share | Dominant isolated stage |
| --- | ---: | ---: | ---: | --- |
| `32768` centroids, `kc120_ef64_alpha0.35` | `1.3555743299023153` | `0.21438380846531743` | `0.6256436175858051` | rerank scoring |
| `262144` centroids, `kc360_ef64_alpha0.35` | `1.58750085175558` | `0.4928831055891983` | `0.3770497727948497` | HNSW traversal |

Artifacts:
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_visited16.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_visited32_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_visited32_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c131072_post_rerank_cleanup_baseline_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c131072_visited32_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/profile_query_policy_c262144_kc360_ef64_alpha0p35_visited32.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/profile_query_policy_c32768_kc120_ef64_alpha0p35_visited32.json`

## Interpretation

Validated:
- `ef*32 + 16` preserves the focused quality gates on the measured bounded rows
- `ef*32 + 16` improves repeated end-to-end QPS on the `32768` and `262144`
  focused rows

Debunked:
- shrinking as far as `ef*16 + 16` is not beneficial on the `262144` stress row
- `ef*32 + 16` is not a universal win; the `131072` eligible row was slightly
  slower in the repeated A/B

Decision:
- keep the `ef*32 + 16` initial visited-table size for HNSW+PQ because it
  improves the current fastest bounded row and the larger stress row, but track
  the `131072` tradeoff explicitly
- do not claim a paper-scale or SOTA result from this bounded local optimization
