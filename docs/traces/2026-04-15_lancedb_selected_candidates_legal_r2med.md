# Trace: Selected LanceDB Indexed Candidates for Legal and R2MED

## Question

After the full indexed sweeps, can we promote a small set of LanceDB indexed
candidates into a practical hard-slice comparison workflow, instead of treating
default indexed LanceDB as the only indexed baseline?

## Why this step was justified

The full sweeps were useful for discovery, but they are too broad to be the
default comparison workflow. A more practical benchmark surface is:

- Kayak exact
- LanceDB scan
- LanceDB indexed default
- one sweep-backed indexed quality candidate
- one sweep-backed indexed tradeoff candidate

That gives a stable, readable scorecard without rerunning a whole grid every
time.

## Code changes

Added:

- `python/kayak_bridge/lancedb_indexed_candidate_bundle.py`
- `python/scripts/bench_lancedb_indexed_candidates.py`
- `python/tests/test_lancedb_indexed_candidate_bundle.py`

The new script:

- benchmarks Kayak exact and LanceDB scan once
- benchmarks any number of explicit LanceDB indexed candidate configs
- freezes each indexed candidate across rebuilds
- writes one multi-system scorecard
- writes one compact candidate bundle that surfaces:
  - best indexed quality candidate
  - fastest indexed candidate that matches or beats scan quality
  - fastest indexed candidate that strictly improves on scan quality

This is intentionally more compact than the full sweep and more informative than
the previous default-only indexed comparison.

## Validation

Targeted tests:

```bash
env PYTHONPATH=python uv run --python 3.11 python -m unittest \
  python.tests.test_lancedb_indexed_candidate_bundle \
  python.tests.test_lancedb_index_sweep \
  python.tests.test_lancedb_lane_a_bundle \
  python.tests.test_task_comparison_bundle \
  python.tests.test_benchmark_variance
```

Result: passed.

## Selected candidates used

These were chosen from the earlier full sweeps.

### Legal

- `default`
- `p8_refine2:index_num_partitions=8,indexed_refine_factor=2`
- `refine2:indexed_refine_factor=2`

Reason:

- `refine2` was the best-quality legal indexed candidate in the sweep
- `p8_refine2` was a strong legal quality-latency tradeoff in the sweep
- `default` stays in the comparison so we can see whether the improved
  candidates materially change the story

### R2MED

- `default`
- `p8_refine2:index_num_partitions=8,indexed_refine_factor=2`
- `refine1:indexed_refine_factor=1`

Reason:

- `refine1` was the best-quality R2MED indexed candidate in the sweep
- `p8_refine2` was the best measured tradeoff candidate in the sweep
- `default` stayed in to test whether it still mattered on rerun

## Legal selected-candidate results

Artifacts:

- `.cache/kayak/legal_rag_bench_real_subset/candidate_compare/legal_rag_bench_candidates_comparison_scorecard.json`
- `.cache/kayak/legal_rag_bench_real_subset/candidate_compare/legal_rag_bench_candidates_candidate_bundle.json`

Baselines:

- Kayak exact: `mrr=0.16666666666666666`, `0.002344019119239723 s`
- LanceDB scan: `mrr=0.16666666666666666`, `0.029998130097131554 s`

Indexed candidates:

- `lancedb_indexed_default`
  - `mrr=0.09836309523809524`
  - `0.024967134390802434 s`
- `lancedb_indexed_refine2`
  - `mrr=0.16916666666666666`
  - `0.026073521157377398 s`
- `lancedb_indexed_p8_refine2`
  - `mrr=0.17551587301587301`
  - `0.03194002219631026 s`

Bundle interpretation:

- best indexed quality candidate:
  - `lancedb_indexed_p8_refine2`
- fastest indexed candidate that matches or beats scan quality:
  - `lancedb_indexed_refine2`
- fastest indexed candidate that strictly improves on scan quality:
  - `lancedb_indexed_refine2`

Legal conclusion:

- the old default indexed baseline is still clearly misleading on this slice
- the selected-candidate workflow surfaces two viable indexed baselines:
  - `refine2` as the faster quality-improving indexed choice
  - `p8_refine2` as the higher-quality indexed choice in this rerun

## R2MED selected-candidate results

Artifacts:

- `.cache/kayak/r2med_biology_real_subset/candidate_compare/r2med_biology_candidates_comparison_scorecard.json`
- `.cache/kayak/r2med_biology_real_subset/candidate_compare/r2med_biology_candidates_candidate_bundle.json`

Baselines:

- Kayak exact: `ndcg=0.8646255554317486`, `0.001346479121518011 s`
- LanceDB scan: `ndcg=0.8646255554317486`, `0.026917800423689187 s`

Indexed candidates:

- `lancedb_indexed_default`
  - `ndcg=0.8815947061932713`
  - `0.020294922924949787 s`
- `lancedb_indexed_p8_refine2`
  - `ndcg=0.878884285398413`
  - `0.023765963212160082 s`
- `lancedb_indexed_refine1`
  - `ndcg=0.8856371782662615`
  - `0.023856727813836187 s`

Bundle interpretation:

- best indexed quality candidate:
  - `lancedb_indexed_refine1`
- fastest indexed candidate that matches or beats scan quality:
  - `lancedb_indexed_default`
- fastest indexed candidate that strictly improves on scan quality:
  - `lancedb_indexed_default`

R2MED conclusion:

- the rerun did **not** support the simplistic claim that `p8_refine2` should
  replace default indexed LanceDB everywhere
- default indexed LanceDB remained the fastest quality-improving indexed choice
  in this selected-candidate rerun
- `refine1` still won on pure quality

## What changed in the interpretation

This is the useful epistemic result of the new workflow:

- the full sweep is for discovering candidate families
- the selected-candidate workflow is for honest recurring comparison

On legal, the workflow confirms that default indexed LanceDB should not be the
only indexed reference.

On R2MED, the workflow shows the opposite risk:

- if we had replaced default with only the sweep winners, we would have hidden a
  rerun where default indexed LanceDB was still the fastest quality-improving
  indexed option

So the correct practical benchmark posture is:

- do not benchmark only default indexed LanceDB on hard slices
- also do not throw default indexed LanceDB away globally
- compare default plus explicit quality/tradeoff candidates side by side

## Practical next step

For future hard-slice LanceDB-vs-Kayak reports, the indexed section should
include:

- default indexed
- one quality-oriented explicit indexed config
- one tradeoff-oriented explicit indexed config

This selected-candidate workflow now provides that directly, without requiring a
full indexed sweep every time.
