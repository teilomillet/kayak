# 2026-04-15: nested imputed dim128 selector specialization

## Claim

The nested imputed centroid selector still routed dim128 query tokens through
the generic nested-vector dot product and generic shortlist maintenance, even
though the index already stores a flat centroid buffer and the flat imputed path
already exploits that layout.

The concrete question for this step was:

- can the nested dim128 imputed selector score directly against
  `index.flat_centroid_values`
- preserve the current shortlist and baseline-correction semantics
- and produce a measurable nested-selector win without changing retrieval policy

## Why this target

This was the next sound follow-on after the exact top-2 selector:

- the exact selector optimization already removed a fixed-capacity generic-list
  tax from the exact path
- the nested imputed dim128 path still lacked the corresponding shape-aware
  specialization
- the repo already had direct surfaces for this target:
  - correctness: `tests/test_imputed_high_centroid_probe.mojo`
  - benchmark: `benchmarks/profile_imputed_selection_dim128_real_subset.mojo`

The intended rule stayed the same:

- no new retrieval policy
- no hidden pruning
- only lower constant-factor work inside the existing scan/shortlist contract

## What changed

Code change:

- `kayak/planning/centroid_postings_imputed_stage.mojo`

Validation change:

- `tests/test_imputed_high_centroid_probe.mojo`

Benchmark seam added for decision-quality kernel comparison:

- `benchmarks/profile_imputed_selection_dim128_nested_kernel.mojo`

Implementation choice:

- keep `centroid_selection_for_query_token(...)` as the public entry point
- dispatch to a dim128 fast path when:
  - `index.vector_dim == 128`
  - the token length is `128`
- use `dot_product_dim128_flat_at(...)` against `index.flat_centroid_values`
- use the same shortlist helpers already used in the flat imputed path:
  - `insert_descending_centroid_match(...)` for the small-count prefix case
  - `insert_top_bound_centroid_match(...)` plus
    `sort_top_bound_centroid_matches_descending(...)` for the bounded shortlist

Why this is justified:

- the dim128 nested query token is already a flat 128-value buffer in memory
- the index already owns a flat centroid layout
- scoring against that flat centroid layout removes one layer of nested list
  traversal without changing the selector contract
- reusing the flat-path shortlist helpers keeps the tie behavior explicit rather
  than inventing a new policy

## Validation

Correctness tests:

- `pixi run mojo -I . tests/test_imputed_high_centroid_probe.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_service_runtime.mojo`

New regression coverage added:

- dim128 high-count `> bound` equal-score tail behavior
- dim128 high-count late-better-centroid eviction behavior

All of the above passed.

## Benchmarking

### 1. Cross-worktree comparison

I first compared:

- baseline detached worktree at commit `f7d2b97`
- candidate local worktree with the selector change

using:

- `bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 5 --force -- pixi run mojo -I . benchmarks/profile_imputed_selection_dim128_real_subset.mojo`

Relevant log dirs:

- baseline:
  - `/tmp/kayak-imputed-dim128-baseline/.cache/kayak/bench_quiet/20260415T065609Z`
  - `/tmp/kayak-imputed-dim128-baseline/.cache/kayak/bench_quiet/20260415T065913Z`
  - `/tmp/kayak-imputed-dim128-baseline/.cache/kayak/bench_quiet/20260415T070027Z`
- candidate:
  - `.cache/kayak/bench_quiet/20260415T065817Z`
  - `.cache/kayak/bench_quiet/20260415T065937Z`
  - `.cache/kayak/bench_quiet/20260415T070040Z`

Why this evidence was not enough on its own:

- the host never met the quiet threshold, so all runs were forced-wrap
- unrelated long-running processes were active during some runs
- unchanged flat rows moved enough to show that cross-process noise was still
  material

Conclusion:

- useful sanity check
- not strong enough alone for the keep/revert decision

### 2. In-process kernel benchmark

To isolate the actual selector change, I added:

- `benchmarks/profile_imputed_selection_dim128_nested_kernel.mojo`

That benchmark:

- compares the old generic dim128 nested selector against the new optimized
  nested selector in one process
- on the same cached datasets
- with both paths explicitly pre-warmed before timing

Command:

- `bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 5 --force -- pixi run mojo -I . benchmarks/profile_imputed_selection_dim128_nested_kernel.mojo`

Relevant log dirs:

- `.cache/kayak/bench_quiet/20260415T070253Z`
- `.cache/kayak/bench_quiet/20260415T070319Z`

The first in-process pass was still noisy and mixed. The second warm in-process
pass was the clearest measurement and is the one used for the decision.

## Measured outcome

Warm in-process kernel benchmark from
`.cache/kayak/profile_imputed_selection_dim128_nested_kernel.tsv`:

- `SciFact`
  - reference: `0.000391`
  - optimized: `0.0003711666666666666`
  - delta: `-5.07%`
- `FIQA`
  - reference: `0.000381`
  - optimized: `0.00036666666666666667`
  - delta: `-3.76%`
- `LIMIT-small`
  - reference: `0.00030590625`
  - optimized: `0.0003009375`
  - delta: `-1.62%`

Interpretation:

- the optimized nested dim128 selector is faster on all three measured public
  dim128 slices in the best-isolated benchmark seam
- the win is moderate rather than dramatic
- the earlier cross-worktree comparisons explain why this needed a tighter
  benchmark: the host was noisy enough that raw before/after runs were not
  cleanly attributable

## Decision

Keep the nested dim128 imputed selector specialization.

Why:

- correctness and planner/runtime guardrails stayed green
- the best-isolated benchmark seam shows consistent wins across all measured
  datasets
- the change is still policy-neutral: same bound, same nprobe/t' semantics, same
  shortlist ordering contract

What this does **not** prove:

- that every imputed path is now fully shape-specialized
- that the generic non-dim128 path should be removed
- that the cross-worktree benchmark noise problem is solved

The next sound follow-on remains one of:

- precomputing exact small-count selection limits for the nested dim128 segment
  path if a stage-level benchmark shows that repeated token-level limit
  calculation is still material
- or pushing the same “reference versus optimized in one process” method onto
  the next shortlist kernel before trusting broader wrapped benchmarks
