# 2026-04-12: `centroid_postings_blockmax` stage and sealed block sidecars

## Why this step

The next sound move after the earlier centroid-head work was to test whether
sealed-segment storage could carry more search-native structure than plain
posting lists.

The specific hypothesis was:

- persist per-centroid block summaries inside `centroid_postings`
- add a real non-default stage-1 generator that can use those summaries
- measure stage-1 faithfulness against exact stage-2 results before doing
  heavier native-engine work

This was intentionally framed as an empirical infra step, not as a claim that
block-pruning would automatically improve the frontier.

## What changed

Storage:

- `CentroidPostingIndex` now carries block metadata:
  - `block_size`
  - `centroid_block_offsets`
  - `block_max_weights`
  - `total_block_count`
- sealed `centroid_postings` artifacts now persist:
  - `centroid_block_offsets.tsv`
  - `block_max_weights.tsv`
  - manifest fields:
    - `block_size`
    - `total_block_count`
- partial block layouts are rejected
- legacy artifacts still load, but `ensure_stored_centroid_posting_index()`
  now upgrades them so the rewritten sidecar is structurally complete
- cache reuse now also checks the block layout version implicitly through
  `block_size`, so changing the default block layout invalidates old sidecars

Generator:

- new explicit stage-1 generator:
  - `centroid_postings_blockmax`
- search plan:
  - `centroid_postings_blockmax_search_plan(...)`
- execution guardrails:
  - requires centroid-postings sidecars
  - requires `posting_order_kind == weight_desc_doc_asc`

Profiling:

- new profile benchmark:
  - `pixi run bench_profile_stage1_blockmax_real_subset_raw`
- output:
  - `.cache/kayak/profile_stage1_blockmax_real_subset.tsv`

## Validation

Tests run:

- `pixi run test_centroid_postings_store`
- `pixi run test_centroid_heads_store`
- `pixi run test_collection_search_plan`
- `pixi run test_storage_compat`

Benchmarks run sequentially, not in parallel, to avoid cross-benchmark
resource contamination:

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

On the light public real-subset stage-1 profile with `candidate_k = 40`:

- `SciFact`
  - `centroid_postings`: `9.19e-05 s`
  - `centroid_postings_blockmax`: `1.134e-04 s`
  - slowdown vs `centroid_postings`: about `+23.4%`
  - mean candidate blocks: `76.67`
  - mean visited blocks: `75.33`
  - mean skipped blocks: `1.33`
  - mean candidate postings: `472.17`
  - mean skipped postings: `3.83`
- `FIQA`
  - `centroid_postings`: `9.57e-05 s`
  - `centroid_postings_blockmax`: `1.221e-04 s`
  - slowdown vs `centroid_postings`: about `+27.5%`
  - mean skipped blocks: `1.67`
  - mean skipped postings: `11.5`
- `LIMIT-small`
  - `centroid_postings`: `8.69e-05 s`
  - `centroid_postings_blockmax`: `1.094e-04 s`
  - slowdown vs `centroid_postings`: about `+25.9%`
  - mean skipped blocks: `0.91`
  - mean skipped postings: `7.0`

Interpretation:

- the sealed block sidecars are working
- the current block-pruning heuristic is active
- but on these public slices it is not pruning enough work to pay for the extra
  bookkeeping

### Candidate-window sweep

Paired against `centroid_postings_head_auto` across the 23 public rows:

- mean candidate-recall delta: `+0.00793`
- mean candidate-generation time delta: about `+2.75e-05 s`

Representative recall gains vs `centroid_postings_head_auto`:

- `BrowseComp-Plus` evidence, `candidate_k = 10`
  - recall delta: `+0.025`
- `BrowseComp-Plus` gold, `candidate_k = 10`
  - recall delta: `+0.025`
- `FIQA`, `candidate_k = 10`
  - recall delta: `+0.03333`

Paired against plain `centroid_postings` across the same 23 rows:

- mean candidate-recall delta: `+0.00290`
- mean candidate-generation time delta: about `+2.73e-05 s`

Interpretation:

- blockmax is a useful explicit candidate-window benchmark point
- it helps some tight-shortlist regimes
- it is still slower than the plain centroid-postings stage

### Vector-budget sweep

Paired against plain `centroid_postings` across the 100 public rows:

- mean candidate-recall delta: `-0.003125`
- mean `nDCG@10` delta: `-0.001119`
- mean recall@k delta: `-0.001354`
- wins / ties / losses on candidate recall: `7 / 76 / 17`

Interpretation:

- blockmax does not dominate `centroid_postings` across the broader compressed
  query / compressed document design space
- the current heuristic therefore does not earn promotion to the main frontier

### Hard-recall stage-aware benchmark

On the current hard-recall public slice:

- against `centroid_postings_head_auto`
  - candidate recall improves at `candidate_k = 10` on both BrowseComp slices
    from `0.375` to `0.4`
  - `nDCG@10` improves at `candidate_k = 10` on both BrowseComp slices
- against plain `centroid_postings`
  - candidate recall is identical on all four paired rows
  - `nDCG@10` is identical on all four paired rows
  - mean search time is worse by about `7.31e-05 s`

Concrete paired rows against `centroid_postings`:

- `BrowseComp-Plus` evidence, `candidate_k = 10`
  - both: candidate recall `0.4`
  - both: `nDCG@10 = 0.26365135965362574`
  - blockmax slower: `0.00059775 s` vs `0.00054325 s`
- `BrowseComp-Plus` gold, `candidate_k = 10`
  - both: candidate recall `0.4`
  - both: `nDCG@10 = 0.26262991308340555`
  - blockmax slower: `0.000588 s` vs `0.00055725 s`

Interpretation:

- on the harder public slice, blockmax mostly closes the gap to
  `centroid_postings_head_auto`
- but it still does not beat plain `centroid_postings`

## Decision

The sound conclusion is:

- keep `centroid_postings_blockmax` as an explicit, benchmarked generator
- keep the sealed block sidecars because they are real native-search structure
  and make future engine work measurable
- do not promote blockmax over plain `centroid_postings`
- do not claim a new Pareto frontier move yet

The evidence says the next heavier work should not be "more policy tuning" on
top of the same posting scan.

The next plausible frontier step is richer native engine work in the WARP / GEM
direction, for example:

- stronger shortlist bounds than per-token local top-`k`
- more compression-aware block or cell layouts
- native traversal that reduces bookkeeping overhead rather than adding it

That is a justified next step because the current measurements show the limits
of the storage-only blockmax layer on today's public slices.
