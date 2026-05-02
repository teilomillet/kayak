# Tachiom HNSW Heap Frontier

Date: `2026-05-02`

## Claim Under Test

The larger-centroid quality-preserving rows are HNSW-traversal heavy. Replacing
linear scans in the HNSW layer frontier with bounded heaps should reduce
traversal cost without changing the selected centroids or final rankings.

Reason:
- the `262144`-centroid eligible row spent most isolated profile time in HNSW
  traversal
- the old layer search scanned the unexpanded candidate list on every expansion
  and maintained the retained best-`ef` set by sorted insertion
- both operations scale poorly when `ef=max(ef_search, k_c)` grows to `360`

## Implementation

Primary files:
- `kayak/search/tachiom_tac_hnsw_pq_dim128.mojo`
- `python/scripts/profile_tachiom_streaming_hnsw_pq_mojo.py`

The HNSW layer search now uses:
- a best-first candidate heap for the unexpanded frontier
- a worst-first retained heap for the best `ef` centroids
- a final heap-to-descending conversion only when the caller takes a prefix of
  the retained centroids
- an `ef == k_c` fast path that returns the retained heap positions directly
  when every retained centroid is consumed

The profile script now accepts the same query-policy overrides as the benchmark:
- `--centroids-per-query-vector`
- `--hnsw-ef-search`
- `--candidate-pruning-alpha`
- `--disable-candidate-pruning`

This was needed so profile rows match the actual quality-gated benchmark rows.
The `ef == k_c` fast path is valid because downstream document score
accumulation treats selected centroids as an order-independent set when no
prefix is discarded.

## Validation

Focused correctness:

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_streaming_index.py
```

Result: `3` tests passed.

The small fixture checks the Mojo HNSW+PQ result against the Python HNSW+PQ
path, so it catches ordering changes in the centroid traversal path.

## Before And After

The first table records the initial heap-frontier change before the later
`ef == k_c` sort skip. The focused refresh below records the current code.

Fixed shape:
- documents: `10135`
- document vectors: `739372`
- queries: `128`
- query vectors: `4096`
- engine: `streaming_tac_hnsw_pq_mojo`
- exact-reference vector cap: `1000000`

Quality floors:
- primary metric at least `0.989`
- final recall@10 vs exact at least `0.8867`

| Artifact | Policy | Before QPS | After QPS | MRR@10 | Final recall@10 vs exact |
| --- | --- | ---: | ---: | ---: | ---: |
| `32768` centroids | `kc120_ef64_alpha0.35` | `69.01278987944805` | `89.25310082533578` | `0.9895833333333334` | `0.8867187500000006` |
| `131072` centroids | `kc180_ef128_alpha0.5` | `40.093095874216985` | `59.539468` | `0.9895833333333334` | `0.8890625000000004` |
| `262144` centroids | `kc360_ef64_alpha0.35` | `24.116374186902767` | `67.76123363711898` | `0.9934895833333334` | `0.8875000000000008` |

Post-optimization narrow sweep:

| Artifact | Best eligible setting | QPS | MRR@10 | Final recall@10 vs exact | Mean candidate window |
| --- | --- | ---: | ---: | ---: | ---: |
| `32768` centroids | `kc120_ef64_alpha0.35` | `91.26208201247302` | `0.9895833333333334` | `0.8867187500000006` | `193.171875` |
| `131072` centroids | `kc200_ef64_alpha0.45` | `67.75599974258095` | `0.9895833333333334` | `0.8906250000000002` | `248.921875` |
| `262144` centroids | `kc360_ef64_alpha0.35` | `72.61081722938026` | `0.9934895833333334` | `0.8875000000000008` | `136.2421875` |

The optimized `32768` baseline remains faster than both larger-centroid
quality-preserving rows on this bounded docs10000 slice.

Focused fast-path refresh:

| Artifact | Policy | Previous heap QPS | Current QPS | MRR@10 | Final recall@10 vs exact |
| --- | --- | ---: | ---: | ---: | ---: |
| `32768` centroids | `kc120_ef64_alpha0.35` | `89.25310082533578` | `91.26208201247302` | `0.9895833333333334` | `0.8867187500000006` |
| `131072` centroids | `kc200_ef64_alpha0.45` | `65.53828788809584` | `67.75599974258095` | `0.9895833333333334` | `0.8906250000000002` |
| `262144` centroids | `kc360_ef64_alpha0.35` | `67.76123363711898` | `72.61081722938026` | `0.9934895833333334` | `0.8875000000000008` |

## Profile Evidence

`262144` centroids, `kc360_ef64_alpha0.35`:

| Stage | Before batch s | Current batch s |
| --- | ---: | ---: |
| full search | `5.274011832250002` | `1.6019103696049843` |
| candidate generation | `4.317804666466665` | `0.8715290408876561` |
| HNSW traversal | `4.353171548300001` | `0.7669984600424934` |
| residual score table | `0.08169743684939938` | `0.08143563385919896` |
| rerank scoring | `0.7015469431704656` | `0.6173423139407548` |

Current focused profiles:

| Artifact | Full search batch s | HNSW traversal share | Rerank scoring share | Dominant isolated stage |
| --- | ---: | ---: | ---: | --- |
| `32768` centroids, `kc120_ef64_alpha0.35` | `1.372639876931591` | `0.21223047984274065` | `0.6327677449457128` | rerank scoring |
| `262144` centroids, `kc360_ef64_alpha0.35` | `1.6019103696049843` | `0.478802356608521` | `0.385378811233356` | HNSW traversal |

Artifacts:
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/profile_query_policy_c262144_kc360_ef64_alpha0p35.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/profile_query_policy_c262144_kc360_ef64_alpha0p35_dual_heap.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/profile_query_policy_c262144_kc360_ef64_alpha0p35_skip_sort.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_narrow_c131072_dual_heap.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_narrow_c262144_dual_heap.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_dual_heap.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/profile_query_policy_c32768_kc120_ef64_alpha0p35_skip_sort.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_skip_sort.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c131072_skip_sort.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_skip_sort.json`

## Interpretation

Validated:
- the heap frontier is a real traversal optimization on the measured Tachiom
  HNSW+PQ path
- quality metrics and exact-overlap gates are unchanged on the focused rows
- larger-centroid quality-preserving rows are now much closer to the baseline
- skipping the final retained-centroid ordering is a measurable win when
  `ef == k_c`

Not validated:
- the larger-centroid rows still do not beat the optimized `32768` baseline
- this remains bounded-slice evidence, not full paper-scale reproduction

Decision:
- keep the heap frontier
- use the post-optimization `32768` row as the current bounded speed baseline
- the next optimization target is split between remaining HNSW traversal and
  residual-PQ rerank scoring, with the `262144` eligible row as the stress case
