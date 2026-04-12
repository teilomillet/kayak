# 2026-04-12: conservative `centroid_postings_head_auto` policy

## Why this step

The fixed-cap `centroid_postings_head` stage was already a legitimate native
baseline, but the posting-cap sweep showed that a single global head size is
not uniformly optimal.

The next sound step was therefore not "replace the fixed baseline" but:

- add an explicit auto-tuned variant
- keep oracle-recall guardrails
- benchmark it against the fixed-cap generator before promoting it

## What was added

New stage-1 generator:

- `centroid_postings_head_auto`

Policy shape:

- start from the existing fixed-cap behavior (`16`)
- only widen the head window on large centroid budgets (`>= 96`)
- widen more aggressively when query vector count is very small (`<= 8`)
- otherwise preserve the existing fixed-cap behavior
- never exceed `candidate_k`
- keep the same weight-sorted centroid-postings contract as the fixed generator

This is intentionally conservative.
The first broader heuristic regressed recall too often, so the final policy is
expansion-only and only in the regime where the measured evidence was cleaner.

## Guardrails

Verified:

- `centroid_postings_head_auto` refuses unordered centroid-postings sidecars
- oracle full-recall tests pass for the same shortlist-size cases as the fixed
  head generator
- exact late interaction remains the stage-2 correctness anchor

Validation command:

- `pixi run test_collection_search_plan`

## Measured outcome

Generated artifacts:

- `.cache/kayak/public_vector_budget_sweep.json`
- `.cache/kayak/public_candidate_window_sweep.json`
- `.cache/kayak/hard_recall_stage_aware_search.json`

### Vector-budget sweep

Paired against `centroid_postings_head` across the 100 public rows:

- mean candidate-recall delta: `+0.000323`
- mean `nDCG@10` delta: `+0.002108`
- wins / ties / losses on candidate recall: `7 / 87 / 6`
- max gain: `+0.05`
- max loss: `-0.034375`

Representative gains:

- `BrowseComp-Plus` evidence, query budget `16`, document budget `128`
  - fixed: recall `0.75`
  - auto: recall `0.80`
- `BrowseComp-Plus` gold, query budget `16`, document budget `128`
  - fixed: recall `0.75`
  - auto: recall `0.80`
- `BrowseComp-Plus` evidence, query budget `8`, document budget `128`
  - fixed: recall `0.75`
  - auto: recall `0.775`

Representative losses:

- `LIMIT-small`, query budget `4`, document budget `128`
  - fixed: recall `0.75625`
  - auto: recall `0.721875`
- `SciFact`, query budget `8`, document budget `128`
  - fixed: recall `0.883333`
  - auto: recall `0.85`

Interpretation:

- the auto policy does not dominate the fixed-cap generator
- it does create an additional useful Pareto point on compressed-query,
  high-centroid-budget runs
- that is enough to keep it as an explicit option, but not enough to silently
  replace the fixed generator

### Candidate-window sweep

On the default non-budgeted candidate-window sweep:

- candidate recall is unchanged on every paired row
- candidate-generation timing changes are small and mixed

Interpretation:

- the auto policy is effectively inert on the default window sweep because it
  mainly activates when query-vector budgets are small

### Hard-recall stage-aware benchmark

On the current hard-recall public slice:

- candidate recall: unchanged on all paired rows
- `nDCG@10`: unchanged on all paired rows

Interpretation:

- the current hard-recall slice does not stress the new policy regime
- that is consistent with the vector-budget evidence above

## Decision

The sound conclusion is:

- keep `centroid_postings_head_auto` as an explicit benchmarked generator
- do not replace `centroid_postings_head`
- use it mainly when testing compressed-query regimes or short natural queries
- keep exact-oracle recall reporting as the acceptance gate for future policy
  changes

This preserves epistemic hygiene:

- the generator exists because it measured slightly better on a real slice of
  the design space
- it is not presented as a universal improvement
- heavier native work should still target a stronger frontier move than this
  policy layer alone achieved
