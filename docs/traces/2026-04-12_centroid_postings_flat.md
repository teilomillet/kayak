# 2026-04-12: `centroid_postings_flat` native stage follow-on

## Why this step

After measuring `centroid_postings_blockmax`, the sound next question was not
"can we add one more heuristic pruning rule?" but rather:

- can a tighter native memory layout reduce stage-1 overhead
- without changing the retrieval semantics
- and without weakening the exact-stage faithfulness checks already in the repo

This was meant as a layout-native follow-on, not as a claim that WARP/GEM-style
engine work is already finished.

## What changed

Storage/index shape:

- `CentroidPostingIndex` now also derives:
  - `flat_centroid_values`
- this is computed once from the validated centroid vectors and kept alongside
  the existing structured view

Stage 1:

- new explicit generator:
  - `centroid_postings_flat`
- new search plan:
  - `centroid_postings_flat_search_plan(...)`
- new scorer:
  - `centroid_posting_flat_scores_for_segment(...)`

Execution:

- `centroid_postings_flat` requires the same sealed `centroid_postings`
  sidecar as plain `centroid_postings`
- it does **not** require a weight-sorted posting order, unlike the head/block
  variants
- for `vector_dim == 128`, it uses the flat dim-128 dot-product path
- otherwise it uses a generic flat-array dot product

Benchmark surface:

- public candidate-window sweep now includes `centroid_postings_flat`
- public vector-budget sweep now includes `centroid_postings_flat`
- hard-recall stage-aware output now includes `centroid_postings_flat`
- the stage-1 profile benchmark also includes it

Task surface:

- a compatibility benchmark alias still exists under the older
  `blockmax`-named task
- a new neutral alias was added:
  - `pixi run bench_profile_stage1_real_subset_raw`

## Guardrails

Tests run:

- `pixi run test_centroid_postings_store`
- `pixi run test_collection_search_plan`

New direct equivalence guardrail:

- `test_centroid_postings_flat_stage_matches_plain_stage_scores()`
- this verifies on the toy centroid-postings fixture that the flat scorer
  returns the same per-document stage-1 scores as plain `centroid_postings`,
  not just the same final reranked winner

Benchmarks were run sequentially, not in parallel, to reduce resource
contamination:

- `pixi run bench_profile_stage1_blockmax_real_subset_raw`
- `pixi run bench_real_subset_vector_budgets`
- `pixi run bench_real_subset_candidate_windows`
- `pixi run bench_hard_recall_real_subset_raw`

Generated artifacts:

- `.cache/kayak/profile_stage1_blockmax_real_subset.tsv`
- `.cache/kayak/public_vector_budget_sweep.json`
- `.cache/kayak/public_candidate_window_sweep.json`
- `.cache/kayak/hard_recall_stage_aware_search.json`

## Measured outcome

### Stage-1 profile

At `candidate_k = 40` on the public real-subset stage-1 profile:

- `SciFact`
  - `centroid_postings`: `9.384831885682265e-05 s`
  - `centroid_postings_flat`: `9.161777331014997e-05 s`
  - delta: `-2.2305455466726804e-06 s`
- `FIQA`
  - `centroid_postings`: `9.883797344252368e-05 s`
  - `centroid_postings_flat`: `9.849662186188749e-05 s`
  - delta: `-3.413515806361886e-07 s`
- `LIMIT-small`
  - `centroid_postings`: `8.56236569962006e-05 s`
  - `centroid_postings_flat`: `8.702615107283243e-05 s`
  - delta: `+1.4024940766318322e-06 s`

Interpretation:

- the flat path is a real latency improvement on two of the three small public
  slices
- it is not a universal win on every slice
- this is therefore evidence of a small but real layout improvement, not yet a
  blanket replacement justification

### Candidate-window sweep

Paired against plain `centroid_postings` across `23` public rows:

- mean candidate-generation time delta:
  `-4.52083333333333e-06 s`
- candidate-recall delta: `0.0`
- recall wins / ties / losses: `0 / 23 / 0`
- time delta sign counts:
  - slower rows: `5`
  - faster rows: `18`

Interpretation:

- on the public candidate-window sweep, `centroid_postings_flat` preserves the
  same candidate recall as `centroid_postings`
- average stage-1 time is slightly lower

### Vector-budget sweep

Paired against plain `centroid_postings` across `100` public rows:

- mean candidate-recall delta: `0.0`
- mean `nDCG@k` delta: `0.0`
- mean recall@k delta: `0.0`
- mean success-rate delta: `0.0`
- every paired row ties on these quality metrics

Interpretation:

- on the public compressed-query / compressed-document design space,
  `centroid_postings_flat` is behaviorally identical to `centroid_postings`

### Hard-recall stage-aware benchmark

Paired against plain `centroid_postings` across the `4` public hard-recall
rows:

- mean candidate-recall delta: `0.0`
- mean `nDCG@10` delta: `0.0`
- mean recall@k delta: `0.0`
- mean success-rate delta: `0.0`
- mean search-time delta: `-3.099999999999997e-05 s`

Interpretation:

- the flat path keeps the same measured faithfulness on the hard public slice
- total search time is slightly better on average

## Decision

The sound conclusion is:

- keep `centroid_postings_flat` as an explicit benchmarked generator
- treat it as the strongest current semantics-preserving native baseline over
  the original `centroid_postings` execution path
- do not silently replace `centroid_postings` everywhere yet, because the
  stage-1 profile is still mixed on individual slices

What this does justify:

- native layout tightening can already buy small speed gains without changing
  candidate recall or judged quality on the current public artifacts
- the next heavier native step should build on this flatter execution shape,
  not on more heuristic sidecars layered over the old one

What it does not justify:

- claiming a large new Pareto move
- claiming WARP/GEM-style native traversal is already implemented

The next sound follow-on is a heavier native engine step that keeps the same
benchmark discipline:

- build on the flatter centroid/query execution layout
- compare against exact-stage candidate recall and judged quality
- only promote a new engine if it moves the measured frontier, not just the
  implementation style
