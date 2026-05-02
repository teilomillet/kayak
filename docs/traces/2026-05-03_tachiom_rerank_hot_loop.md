# Tachiom Rerank Hot Loop

Date: `2026-05-03`

## Claim Under Test

The post-sort-skip `32768` row is residual-PQ rerank dominated. Small rerank
hot-loop changes should be kept only if they preserve the quality gates and
survive direct A/B measurement on both the `32768` baseline row and the
`262144` larger-centroid stress row.

Reason:
- the `32768` profile after the HNSW heap and sort-skip work had rerank scoring
  at `0.6327677449457128` of full-search isolated time
- the `262144` profile remained split between HNSW traversal and rerank scoring,
  so a rerank-only change must not regress that stress row
- benchmark variance is material enough that a single exploratory run should
  not be used as the only decision point for a small optimization

## Changes Tested

Rejected:
- on-demand centroid-score caching in rerank
- address-backed engine promotion for this row
- inlining the token scoring body inside the document max scan

Kept:
- remove the redundant `token_index == start_token` branch in the rerank
  document-token max scan
- reserve the small rerank-score and winner buffers before appending

Reason:
- `min_score_scalar()` is below any valid score, so the first token does not
  need a separate branch
- `candidate_positions` and winner counts are already known before appending
- the change does not alter scores, candidate sets, vector counts, or search
  policy

## Validation

Focused correctness:

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_streaming_index.py
```

Result: `3` tests passed.

The Mojo fixture compares normal and address-backed HNSW+PQ candidate and final
rankings against the Python path on a dim128 fixture.

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
- quiet wrapper with `sweep_repeats=3` for the A/B rows

### Controlled A/B

| Artifact | Policy | Baseline QPS | Cleanup QPS | MRR@10 | Final recall@10 vs exact |
| --- | --- | ---: | ---: | ---: | ---: |
| `32768` centroids | `kc120_ef64_alpha0.35` | `91.72207055646975` | `92.33851822318282` | `0.9895833333333334` | `0.8867187500000006` |
| `262144` centroids | `kc360_ef64_alpha0.35` | `70.7116733022006` | `70.85580779445326` | `0.9934895833333334` | `0.8875000000000008` |

The measured gain is small. It is enough to keep the cleanup because the code is
simpler and did not regress the larger-centroid stress row in the controlled
A/B, but it is not a new paper-reproduction result.

### Debunked Alternatives

| Attempt | Row | QPS | Result |
| --- | --- | ---: | --- |
| centroid-score cache | `32768`, `kc120_ef64_alpha0.35` | `83.682714` | rejected; cache overhead exceeded saved centroid-dot work |
| address-backed engine | `32768`, `kc120_ef64_alpha0.35` | `47.843980` | rejected; pointer-backed path was slower on this row |
| token-scoring inline | `32768`, `kc120_ef64_alpha0.35` | `91.835924` | rejected; lower than the smaller cleanup on the same row |
| token-scoring inline | `262144`, `kc360_ef64_alpha0.35` | `70.266584` | rejected; slower than the controlled baseline |

Artifacts:
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_rerank_centroid_cache.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_address.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_rerank_inline.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_rerank_inline.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_post_skip_sort_baseline_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_post_skip_sort_baseline_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_rerank_cleanup_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_rerank_cleanup_repeats3.json`

## Interpretation

Validated:
- the kept cleanup preserves judged MRR and final recall@10 vs exact on the two
  focused rows
- the kept cleanup is a small controlled improvement, not a structural
  rerank-speed breakthrough

Debunked:
- centroid-score caching is not beneficial on the measured bounded row
- the address-backed engine should not replace the regular Mojo engine for this
  benchmark row
- inlining token scoring is not an improvement on the stress row

Decision:
- keep the small hot-loop cleanup
- keep the stronger HNSW heap and sort-skip rows as the main current speedups
- continue treating full paper-scale throughput and SOTA as not reproduced
