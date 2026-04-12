# Hard-Recall Stage-Aware Benchmark

Date: April 12, 2026

## Goal

Verify that Kayak can now report:
- stage-1 candidate recall against an exact full-scan reference
- final retrieval quality after exact rescoring
- latency and storage context in the same benchmark output

This trace is specifically for the hard-recall benchmark entrypoint:

```bash
pixi run bench_hard_recall_real_subset_raw
```

Output:

```text
.cache/kayak/hard_recall_stage_aware_search.json
```

## Why These Slices

This benchmark uses:
- BrowseComp-Plus evidence slice
- BrowseComp-Plus gold slice

It intentionally does not include `LIMIT-small`.

Reason:
- existing repo measurements already show `LIMIT-small` is useful as a compact
  regression slice
- existing repo measurements do not show it as the strongest stage-1 recall
  stress test

That selection logic is recorded in:
- [docs/hard_recall_evaluation.md](../hard_recall_evaluation.md)

## Measured Results

This run covered:
- `exact_full_scan`
- `document_proxy`
- `centroid_heads`
- `centroid_postings`
- `centroid_postings_head`
- `centroid_postings_imputed`

The non-exact plans used:
- `faithfulness_policy_kind = "best_effort"`

### BrowseComp-Plus Evidence Slice

- exact full scan, `candidate_k=10`
  - `mean_ndcg@10 = 0.2623`
  - `mean_candidate_recall_at_final_k = 1.0`
  - `mean_search_seconds = 0.001249`
- document proxy, `candidate_k=10`
  - `mean_ndcg@10 = 0.1700`
  - `mean_candidate_recall_at_final_k = 0.725`
  - `mean_search_seconds = 0.0004225`
- centroid heads, `candidate_k=10`
  - `mean_ndcg@10 = 0.2637`
  - `mean_candidate_recall_at_final_k = 0.4`
  - `mean_search_seconds = 0.00057625`
- centroid postings, `candidate_k=10`
  - `mean_ndcg@10 = 0.2637`
  - `mean_candidate_recall_at_final_k = 0.4`
  - `mean_search_seconds = 0.00060925`
- centroid postings head, `candidate_k=10`
  - `mean_ndcg@10 = 0.2023`
  - `mean_candidate_recall_at_final_k = 0.375`
  - `mean_search_seconds = 0.00052275`
- centroid postings imputed, `candidate_k=10`
  - `mean_ndcg@10 = 0.3218`
  - `mean_candidate_recall_at_final_k = 0.4`
  - `mean_search_seconds = 0.0008715`
- all non-exact plans, `candidate_k=90`
  - `mean_candidate_recall_at_final_k = 1.0`
  - `mean_ndcg@10 = 0.2623`
  - `mean_search_seconds ≈ 0.00464` to `0.00499`

Interpretation:
- at a tight `candidate_k=10`, the current public stage-1 families lose
  exact-reference recall in different ways
- `document_proxy` preserves more of the exact top-`10` than the current
  centroid families on this slice
- all evaluated stage-1 families recover the exact full-scan result once the
  candidate budget expands to the full 90-document slice
- `centroid_postings_imputed` outperforms exact full scan on judged `nDCG@10`
  here despite lower candidate recall against the exact reference; that is not
  a contradiction because judged ranking quality and recall against the exact
  full-scan top-`k` are different measurements

### BrowseComp-Plus Gold Slice

- exact full scan, `candidate_k=10`
  - `mean_ndcg@10 = 0.2851`
  - `mean_candidate_recall_at_final_k = 1.0`
  - `mean_search_seconds = 0.001186`
- document proxy, `candidate_k=10`
  - `mean_ndcg@10 = 0.1736`
  - `mean_candidate_recall_at_final_k = 0.725`
  - `mean_search_seconds = 0.0004495`
- centroid heads, `candidate_k=10`
  - `mean_ndcg@10 = 0.2626`
  - `mean_candidate_recall_at_final_k = 0.4`
  - `mean_search_seconds = 0.00058325`
- centroid postings, `candidate_k=10`
  - `mean_ndcg@10 = 0.2626`
  - `mean_candidate_recall_at_final_k = 0.4`
  - `mean_search_seconds = 0.0005545`
- centroid postings head, `candidate_k=10`
  - `mean_ndcg@10 = 0.2484`
  - `mean_candidate_recall_at_final_k = 0.375`
  - `mean_search_seconds = 0.0005345`
- centroid postings imputed, `candidate_k=10`
  - `mean_ndcg@10 = 0.4450`
  - `mean_candidate_recall_at_final_k = 0.4`
  - `mean_search_seconds = 0.00085875`
- all non-exact plans, `candidate_k=90`
  - `mean_candidate_recall_at_final_k = 1.0`
  - `mean_ndcg@10 = 0.2851`
  - `mean_search_seconds ≈ 0.00479` to `0.00514`

Interpretation:
- the gold slice shows the same exact-reference recall pressure at tight
  candidate budgets
- `centroid_postings_imputed` again improves judged `nDCG@10` relative to exact
  full scan while still losing exact-reference recall at `candidate_k=10`
- the current benchmark therefore needs both measurements:
  judged retrieval quality and candidate recall against the exact reference

## Storage Context

The benchmark output recorded:
- `document_count = 90`
- `token_count = 15756`
- `vector_count = 15756`
- `byte_size ≈ 8.28 MB`
- `bytes_per_document ≈ 91.95 KB`
- `bytes_per_vector ≈ 525.21 bytes`

That storage context is larger than the earlier exact-only mirror because the
current collection mirror includes the sidecars needed by the public stage-1
families.

## Evidence Summary

Verified after the benchmark run:
- Kayak can run an explicit approximate stage 1 plus exact stage 2 search path
- Kayak can report candidate recall against an exact full-scan reference
- Kayak can write a stage-aware benchmark artifact on harder public slices

Not verified here:
- comparison against a long-context LLM or cross-encoder ceiling
- a production-worthy approximate generator beyond the current public baselines
