# 2026-04-15: exact top-2 centroid selector specialization

## Claim

The exact centroid stage still paid generic `List` insertion costs even though
its contract is always "keep the best 2 centroids per query token."

The concrete question for this step was:

- can we replace the exact top-2 generic insertion path with a fixed-capacity
  kernel
- preserve the current stable tie behavior exactly
- and produce a measurable stage-1 selector win without changing retrieval
  policy

## Why this target

This was the next sound target after the sparse accumulator work:

- the accumulator split did not clear the evidence bar under wrapped benchmarks
  and was reverted
- the exact centroid selector still scanned all centroids, so the only remaining
  Mojo-only lever there was constant-factor reduction
- the benchmark harness already isolates `exact_selection_nested` and
  `exact_selection_flat` in
  `benchmarks/profile_centroid_primitives_real_subset.mojo`

That made this a good kernel optimization target:

- local implementation surface
- stable semantics
- direct benchmark visibility

## What changed

Code changes:

- `kayak/planning/centroid_postings_stage.mojo`
- `kayak/planning/centroid_postings_flat_stage.mojo`

Validation guardrail:

- `tests/test_centroid_exact_selection.mojo`

Implementation choice:

- keep the generic `insert_descending_centroid_match(...)` helper for imputed
  and higher-`k` callers
- add a dedicated exact top-2 helper that stores only two `(index, score)`
  slots
- materialize the final `ScoredCentroidSelection` once after the centroid scan

Why this is justified:

- exact centroid selection is fixed at `k = 2`, so dynamic insertion and
  per-centroid `List` traffic are avoidable
- the optimization is policy-neutral because it preserves the same "scan all
  centroids, keep best two" contract
- the helper explicitly preserves the previous stable tie behavior:
  - keep the earlier tail entry on equal scores
  - replace a weaker second slot when a later centroid ties the current best

## Validation

Correctness tests:

- `pixi run mojo -I . tests/test_centroid_exact_selection.mojo`
- `pixi run mojo -I . tests/test_collection_search_plan.mojo`
- `pixi run mojo -I . tests/test_service_runtime.mojo`

Benchmark command used for the final keep/revert decision:

- baseline worktree at commit `43e9a66`
- baseline command:
  - `bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 5 --force -- pixi run bench_profile_centroid_primitives_real_subset_raw`
- candidate command:
  - `bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 5 --force -- pixi run bench_profile_centroid_primitives_real_subset_raw`

Artifacts:

- baseline log dir:
  - `/tmp/kayak-top2-baseline/.cache/kayak/bench_quiet/20260415T061041Z`
- candidate log dir:
  - `.cache/kayak/bench_quiet/20260415T062853Z`
- baseline TSV:
  - `/tmp/kayak-top2-baseline/.cache/kayak/profile_centroid_primitives_real_subset.tsv`
- candidate TSV:
  - `.cache/kayak/profile_centroid_primitives_real_subset.tsv`

Important caveat:

- the host never satisfied the normal quiet threshold, so the wrapper had to run
  in `--force` mode after a short timeout
- this is therefore warm forced-wrap evidence, not quiet-host median-quality
  evidence
- I still prefer it over raw single-shot timings because the same wrapper,
  command, and host constraints were used for both baseline and candidate

## Measured outcome

Exact selector delta, candidate versus warm baseline:

- `SciFact`
  - `exact_selection_nested`: `0.00011134780089798677 -> 7.884726896194668e-05`
    (`-29.19%`)
  - `exact_selection_flat`: `8.82875014073583e-05 -> 7.705978623146765e-05`
    (`-12.72%`)
- `FIQA`
  - `exact_selection_nested`: `0.00011776285048774936 -> 9.678235049833887e-05`
    (`-17.82%`)
  - `exact_selection_flat`: `8.412906861875934e-05 -> 8.751413436897308e-05`
    (`+4.02%`)
- `LIMIT-small`
  - `exact_selection_nested`: `0.00010827099929259066 -> 8.221006637168141e-05`
    (`-24.07%`)
  - `exact_selection_flat`: `8.518295998326999e-05 -> 7.72112095824967e-05`
    (`-9.36%`)

Interpretation:

- the nested exact selector win is clear across all three datasets
- the flat exact selector wins on two of three datasets and shows one small
  regression on FIQA
- the mixed flat result means this is not evidence for a blanket "all exact
  selection got faster everywhere" claim
- it is strong enough evidence to keep the change because:
  - the main nested path improved consistently and materially
  - the flat path improved in two datasets by more than the one FIQA regression
  - correctness guardrails stayed green

## Decision

Keep the exact top-2 specialization.

Why:

- it removes avoidable generic-container work from the exact selector hot path
- it preserves the existing top-2 semantics under direct reference tests
- it produced real benchmark wins on the nested exact selector across all
  measured datasets

What this does **not** prove:

- that the flat exact path is uniformly faster on every slice
- that centroid scanning itself is no longer the dominant asymptotic cost
- that the next imputed or accumulation optimization should automatically be
  accepted without the same benchmark discipline

The next sound follow-on remains:

- a shape-specialized imputed selector for the nested dim128 path, or
- a benchmark-backed revisit of the flat exact selector if FIQA-like regressions
  become a recurring pattern under quieter runs
