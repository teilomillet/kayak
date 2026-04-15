# 2026-04-15: planner wide-window `centroid_postings_flat` crossover

## Claim

For `balanced` and `native_multivector`, the planner should stop preferring
`centroid_postings_imputed_flat` once the candidate window is already wide
enough to saturate candidate recall.

The concrete claim tested here was:

- when `candidate_k >= final_k * 8`
- and the goal is `balanced` or `native_multivector`
- `centroid_postings_flat` is the better default candidate generator order
  than `centroid_postings_imputed_flat`

Why this needed verification:

- the earlier gold planner artifact suggested the crossover
- but the planner did not yet encode it
- and a planner-order change is only justified if measured latency improves
  without giving back judged quality or candidate recall

## Why this target

The previous head-auto retune closed the small-window long-query gap.

The next measurable gap was different:

- hard small windows still favored `centroid_postings_head_auto` or
  `centroid_postings_imputed_flat`
- but the refreshed gold planner artifact showed the wide windows (`80`, `90`)
  selecting `imputed_flat` even though those rows looked latency-dominated
  relative to `centroid_postings_flat`

This made planner ordering the sound next target rather than another scoring
kernel edit.

## What changed

Planner rule:

- `kayak/planning/planner.mojo`

New planner guardrails:

- `tests/test_search_planner.mojo`
- `tests/test_planner_benchmark_json.mojo`
- `tests/test_planner_evidence_json.mojo`

New benchmark seams:

- `benchmarks/public_small_window_frontier.mojo`
- `benchmarks/public_wide_window_native_frontier.mojo`
- `benchmarks/public_wide_window_candidate_generation.mojo`

Task wiring:

- `pyproject.toml`

The planner change is intentionally narrow:

- keep the existing goal-default order for normal windows
- only for `candidate_k >= final_k * 8`
- and only for `balanced` / `native_multivector`
- move `centroid_postings_flat` ahead of `centroid_postings_imputed_flat`

Why this rule shape is justified:

- the planner cannot yet see query vector count, so this is the largest
  reliable signal it already has
- the small-window public frontier showed `imputed_flat` and `head_auto` are
  still important there
- the wide-window measurements below show the opposite regime

## Validation

Planner tests:

- `pixi run bash -lc 'mojo -I . tests/test_search_planner.mojo'`
- `pixi run bash -lc 'mojo -I . tests/test_planner_benchmark_json.mojo'`
- `pixi run bash -lc 'mojo -I . tests/test_planner_evidence_json.mojo'`

Planner benchmark refresh:

- `pixi run bench_real_subset_planner_benchmark_raw`

Focused frontiers:

- `pixi run bash -lc 'mojo -I . benchmarks/public_small_window_frontier.mojo'`
- `pixi run bash -lc 'mojo -I . benchmarks/public_wide_window_native_frontier.mojo'`
- `pixi run bash -lc 'mojo -I . benchmarks/public_wide_window_candidate_generation.mojo'`

Artifacts:

- `.cache/kayak/public_planner_benchmark.json`
- `.cache/kayak/public_small_window_frontier.json`
- `.cache/kayak/public_wide_window_native_frontier.json`
- `.cache/kayak/public_wide_window_candidate_generation.json`

Quiet-wrapper attempt:

- `pixi run bench_public_wide_window_native_frontier`

Quiet-wrapper artifact:

- `.cache/kayak/bench_quiet/20260415T055330Z`

Result of the quiet attempt:

- it timed out after `120s`
- the host never reached the default quiet threshold because unrelated
  background CPU stayed between roughly `556` and `765`
- that means the quiet wrapper could not produce decision-quality whole-search
  timing on this machine during this run

This matters because it limits how much weight we should put on raw
whole-search latency deltas.

## Measured outcome

### Public planner benchmark after the rule

The refreshed planner artifact now flips only the wide windows:

- `fiqa_real_subset`
  - `balanced`: `80 -> centroid_postings_flat`, `82 -> centroid_postings_flat`
  - `native_multivector`: `80 -> centroid_postings_flat`,
    `82 -> centroid_postings_flat`
- `browsecomp_plus_evidence_slice`
  - `balanced`: `80 -> centroid_postings_flat`, `90 -> centroid_postings_flat`
  - `native_multivector`: `80 -> centroid_postings_flat`,
    `90 -> centroid_postings_flat`
- `browsecomp_plus_gold_slice`
  - `balanced`: `80 -> centroid_postings_flat`, `90 -> centroid_postings_flat`
  - `native_multivector`: `80 -> centroid_postings_flat`,
    `90 -> centroid_postings_flat`

Small windows stayed unchanged:

- `10`, `20`, `40` continue to select `centroid_postings_imputed_flat`
  for `balanced` and `native_multivector`

This matches the intended planner shape exactly.

### Small-window frontier check

The public small-window frontier justified **not** changing the planner more
aggressively:

- `head_auto` remained strong on the hard BrowseComp slices
- easier datasets still often favored `document_proxy` or the existing
  `imputed_flat` ordering in the smaller windows

So the new rule does **not** claim `flat` is universally better.

### Whole-search wide-window check

The raw whole-search wide-window frontier showed that quality and candidate
recall were matched at the wide windows:

- `fiqa_real_subset`
  - `k = 80`: both `flat` and `imputed_flat` had `candidate_recall = 1.0`
    and `nDCG@10 = 0.9942017072368796`
  - `k = 82`: both had `candidate_recall = 1.0`
    and `nDCG@10 = 0.9942017072368796`
- `browsecomp_plus_evidence_slice`
  - `k = 80`: both had `candidate_recall = 0.975`
    and `nDCG@10 = 0.28512072045816483`
  - `k = 90`: both had `candidate_recall = 1.0`
    and `nDCG@10 = 0.26234761965070796`
- `browsecomp_plus_gold_slice`
  - `k = 80`: both had `candidate_recall = 0.975`
    and `nDCG@10 = 0.31903977127133754`
  - `k = 90`: both had `candidate_recall = 1.0`
    and `nDCG@10 = 0.2851267779084149`

Interpretation:

- by the time the planner reaches these windows, `imputed_flat` is not buying
  better candidate recall or better judged quality
- the only remaining question is latency

### Candidate-generation microbenchmark

The decisive measurement came from the new stage-1 benchmark, which isolates
candidate generation and repeats the query loop `64x` inside the benchmark
runner to reduce timing noise.

Measured rows from
`.cache/kayak/public_wide_window_candidate_generation.json`:

- `fiqa_real_subset`
  - `k = 80`
    - `flat`: `0.00008443773744195863s`, recall `1.0`
    - `imputed_flat`: `0.00043875877192982456s`, recall `1.0`
    - ratio: `5.196x`
  - `k = 82`
    - `flat`: `0.00008609987085665087s`, recall `1.0`
    - `imputed_flat`: `0.00043745851528384283s`, recall `1.0`
    - ratio: `5.081x`
- `browsecomp_plus_evidence_slice`
  - `k = 80`
    - `flat`: `0.000089028927458834s`, recall `0.975`
    - `imputed_flat`: `0.00044288053097345134s`, recall `0.975`
    - ratio: `4.975x`
  - `k = 90`
    - `flat`: `0.00008902403204272362s`, recall `1.0`
    - `imputed_flat`: `0.0004372336244541484s`, recall `1.0`
    - ratio: `4.911x`
- `browsecomp_plus_gold_slice`
  - `k = 80`
    - `flat`: `0.00008788093145869947s`, recall `0.975`
    - `imputed_flat`: `0.00043614379084967324s`, recall `0.975`
    - ratio: `4.963x`
  - `k = 90`
    - `flat`: `0.00008833966431095407s`, recall `1.0`
    - `imputed_flat`: `0.00044108149779735684s`, recall `1.0`
    - ratio: `4.993x`

Interpretation:

- in the exact wide-window regime the planner rule targets
- `centroid_postings_flat` preserves the same candidate recall as
  `centroid_postings_imputed_flat`
- while reducing candidate-generation latency by roughly `5x`

That is strong enough to justify the planner crossover even though the
whole-search quiet wrapper could not run successfully on this host.

## Code-path explanation

The measurements match the code.

`centroid_postings_flat_stage.mojo` does the simpler path:

- top-2 centroid selection per query token
- immediate accumulation into the segment workspace

`centroid_postings_imputed_flat_stage.mojo` does extra work:

- bounded shortlist maintenance
- shortlist sorting / finalization
- baseline-correction accumulation
- storing selections before the final accumulation pass

At narrow windows that extra work can help preserve recall.
At the wide windows measured above, recall is already saturated, so that extra
work becomes overhead rather than benefit.

## Decision

Keep the wide-window planner crossover.

Why this is justified:

- the planner benchmark now flips only the intended rows
- the small-window frontier says not to widen the change further
- the whole-search benchmark shows matched quality / recall at the wide windows
- the stage-1 repeated microbenchmark shows a consistent `~5x` latency win for
  `centroid_postings_flat`
- the code paths explain why that crossover exists

What this does **not** justify:

- making `flat` the default at small windows
- claiming whole-search end-to-end latency is exactly `5x` better on a noisy
  host
- promoting `head_auto` globally without first plumbing query-vector count into
  the planner

The next sound follow-on is:

- make query-vector count available to the planner
- then test whether long-query hard slices should explicitly route to
  `centroid_postings_head_auto` at small windows while keeping the new
  wide-window `flat` crossover
