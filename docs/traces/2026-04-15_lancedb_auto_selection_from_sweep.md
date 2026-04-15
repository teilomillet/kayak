# Trace: Automatic LanceDB Indexed Candidate Selection from Sweep Summaries

## Question

Can we remove the last manual step in the LanceDB hard-slice workflow by
selecting candidate configs directly from the indexed sweep summary, then rerun
those selected candidates without hand-picking `refine1`, `p8_refine2`, or
other configs by name?

## Why this step was justified

Before this change, the workflow had two strong pieces:

- full indexed sweeps to discover good LanceDB settings
- selected-candidate reruns to keep later comparisons compact

But one important step was still manual:

- a human had to choose which sweep rows became "selected candidates"

That meant the workflow was still partly dependent on judgment in the terminal
history instead of on a recorded policy.

## Code changes

Added:

- `python/kayak_bridge/lancedb_indexed_selection.py`
- `python/tests/test_lancedb_indexed_selection.py`

Updated:

- `python/scripts/bench_lancedb_indexed_candidates.py`

The candidate benchmark script now supports:

- `--candidate-config ...` for manual explicit configs
- `--sweep-summary path/to/summary.json` for automatic selection from an
  existing indexed sweep

When `--sweep-summary` is used, the script now writes:

- the usual scorecard and candidate bundle
- one machine-readable selection artifact:
  - `*_candidate_selection.json`

## Selection policy

The policy is intentionally simple and explicit:

1. include `default` when present
2. include the best-quality indexed config from the sweep
3. include the fastest indexed config that strictly improves on scan quality

This is recorded verbatim in the selection artifact as:

- `include_default_plus_best_quality_plus_fastest_quality_improving`

I chose this policy because it is easy to audit from the sweep JSON and does not
try to hide subjective weighting behind a more complicated formula.

## Validation

Targeted tests:

```bash
env PYTHONPATH=python uv run --python 3.11 python -m unittest \
  python.tests.test_lancedb_indexed_selection \
  python.tests.test_lancedb_indexed_candidate_bundle \
  python.tests.test_lancedb_index_sweep \
  python.tests.test_benchmark_variance
```

Result: passed.

## What the selector chose

### Legal

Input sweep:

- `.cache/kayak/legal_rag_bench_real_subset/index_sweep_full/legal_rag_bench_index_sweep_indexed_sweep_summary.json`

Selected configs:

- `default`
- `refine2`
- `sv16_refine2`

Artifact:

- `.cache/kayak/legal_rag_bench_real_subset/candidate_compare_auto/legal_rag_bench_candidates_auto_candidate_selection.json`

Interpretation:

- `refine2` was the sweep winner on pure quality
- `sv16_refine2` was the fastest sweep config that still beat scan quality
- this is more reproducible than hand-picking `p8_refine2` or another tradeoff
  config by inspection

### R2MED

Input sweep:

- `.cache/kayak/r2med_biology_real_subset/index_sweep_full/r2med_biology_index_sweep_indexed_sweep_summary.json`

Selected configs:

- `default`
- `refine1`
- `p8_refine2`

Artifact:

- `.cache/kayak/r2med_biology_real_subset/candidate_compare_auto/r2med_biology_candidates_auto_candidate_selection.json`

Interpretation:

- `refine1` was the sweep winner on pure quality
- `p8_refine2` was the fastest sweep config that still improved on scan quality
- default remained included separately because the policy always keeps it when
  present

## End-to-end reruns

I then reran the selected-candidate benchmark in automatic mode.

### Legal auto-selected rerun

Artifacts:

- `.cache/kayak/legal_rag_bench_real_subset/candidate_compare_auto/legal_rag_bench_candidates_auto_candidate_bundle.json`
- `.cache/kayak/legal_rag_bench_real_subset/candidate_compare_auto/legal_rag_bench_candidates_auto_comparison_scorecard.json`

Baselines:

- Kayak exact: `mrr=0.16666666666666666`, `0.001029782957630232 s`
- LanceDB scan: `mrr=0.16666666666666666`, `0.028432784713610697 s`

Selected indexed candidates:

- default
  - `mrr=0.12375`
  - `0.021997532244616497 s`
- refine2
  - `mrr=0.17506944444444444`
  - `0.026983602476927143 s`
- sv16_refine2
  - `mrr=0.16979166666666665`
  - `0.025442706659669058 s`

Bundle interpretation:

- best indexed quality candidate:
  - `lancedb_indexed_refine2`
- fastest indexed candidate that strictly improves on scan quality:
  - `lancedb_indexed_sv16_refine2`

Legal conclusion:

- the automatic selector produced a useful and defensible compact benchmark set
- it surfaced the same broad pattern as the full sweep:
  - default is weak on quality
  - refine2 wins on indexed quality
  - a lower-cost quality-improving config exists and is worth carrying forward

### R2MED auto-selected rerun

Artifacts:

- `.cache/kayak/r2med_biology_real_subset/candidate_compare_auto/r2med_biology_candidates_auto_candidate_bundle.json`
- `.cache/kayak/r2med_biology_real_subset/candidate_compare_auto/r2med_biology_candidates_auto_comparison_scorecard.json`

Baselines:

- Kayak exact: `ndcg=0.8646255554317486`, `0.0012182309195244063 s`
- LanceDB scan: `ndcg=0.8646255554317486`, `0.026781451369364124 s`

Selected indexed candidates:

- default
  - `ndcg=0.8693617337449628`
  - `0.02030327117148166 s`
- refine1
  - `ndcg=0.881970337132496`
  - `0.021117255938588642 s`
- p8_refine2
  - `ndcg=0.8736968207186862`
  - `0.023976183701112555 s`

Bundle interpretation:

- best indexed quality candidate:
  - `lancedb_indexed_refine1`
- fastest indexed candidate that strictly improves on scan quality:
  - `lancedb_indexed_default`

R2MED conclusion:

- the selector chose `p8_refine2` as the sweep-time "fastest quality-improving"
  config
- the direct rerun still showed `default` as the fastest quality-improving
  indexed option among the selected set

This is not a failure of the workflow. It is a useful result:

- automatic selection removes hand curation
- the rerun bundle still tells us what actually won under the new run

## Practical conclusion

This closes the last manual gap in the workflow:

1. run the broad indexed sweep
2. automatically select candidate configs from the sweep summary
3. rerun only those selected candidates in a compact comparison

The important epistemic point is that selection and rerun are now separated:

- the selection policy is explicit and reproducible
- the final bundle still reports what actually won on rerun

## Next implication

For future hard-slice LanceDB-vs-Kayak work, the default path no longer needs a
human to say "use `p8_refine2` here". The repo can derive that candidate set
directly from the sweep artifact, while still checking the rerun outcome rather
than assuming the sweep ordering will always remain unchanged.
