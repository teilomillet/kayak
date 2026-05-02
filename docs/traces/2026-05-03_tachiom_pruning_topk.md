# Tachiom Pruning-Aware Top-K

Date: `2026-05-03`

## Claim Under Test

Candidate pruning keeps far fewer documents than the configured `candidate_k`
on the measured HNSW+PQ rows, so ranking every first-stage score into a
`candidate_k=1000` heap may do avoidable work.

Reason:
- the measured candidate windows after pruning are much smaller than `1000`
- candidate pruning defines a score threshold from the final-rank cutoff
- if the cutoff is known first, scores below that threshold cannot survive the
  pruned candidate window

## Change

Kept:
- add `pruned_top_positions_by_score(...)`
- compute the final-rank cutoff first
- scan all document scores and heap-rank only positions that either pass the
  pruning threshold or are required to preserve the first `final_k` candidates

Reason:
- it preserves the existing candidate-pruning rule, including the edge case
  where negative scores still retain the first `final_k` candidates
- it does not change vector counts, HNSW traversal, score accumulation, PQ
  residual scoring, or exact rerank semantics
- it reduces heap pressure when the pruning threshold removes most candidate
  positions

Rejected:
- sparse candidate top-k over only seen document positions

Reason:
- the fixture passed, but the `262144` stress row dropped to `72.00605038945794`
  QPS versus the committed `73.48173692089176` QPS baseline
- the isolated profile showed candidate top-k was not a large enough bottleneck
  to justify extra score-accumulation bookkeeping

## Validation

Focused correctness:

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_streaming_index.py
```

Result: `3` tests passed.

The fixture compares Python, list-backed Mojo, and address-backed Mojo candidate
positions and final rankings on a dim128 HNSW+PQ fixture.

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

| Artifact | Policy | Previous QPS | Pruning-aware top-k QPS | MRR@10 | Final recall@10 vs exact | Mean pruned candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `32768` centroids | `kc120_ef64_alpha0.35` | `92.78163789778101` | `94.92727410156428` | `0.9895833333333334` | `0.8867187500000006` | `193.171875` |
| `131072` centroids | `kc200_ef64_alpha0.45` | `68.18820304691724` | `69.5078171936609` | `0.9895833333333334` | `0.8906250000000002` | `248.921875` |
| `262144` centroids | `kc360_ef64_alpha0.35` | `73.48173692089176` | `74.9128485720521` | `0.9934895833333334` | `0.8875000000000008` | `136.2421875` |

Artifacts:
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/query_policy_focus_c32768_pruned_topk_guard_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c131072_pruned_topk_guard_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_pruned_topk_guard_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/query_policy_focus_c262144_sparse_topk_repeats3.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_centroid_ladder/profile_c262144_sparse_topk.json`

## Interpretation

Validated:
- pruning-aware top-k preserves the focused MRR and final recall gates on the
  three measured bounded rows
- it improves repeated quiet-wrapper QPS on the current `32768`, `131072`, and
  `262144` quality-preserving rows

Debunked:
- tracking sparse seen-document positions is not beneficial on the measured
  `262144` stress row

Decision:
- keep pruning-aware top-k as a bounded local optimization
- continue treating paper-scale throughput, full MS MARCO reproduction, and SOTA
  as not reproduced
