# 2026-04-15 R2MED Biology Late-Interaction Operator Sweep

## Question

Can Kayak improve pure late-interaction quality on the full `R2MED/Biology`
benchmark by changing the token-pooling operator instead of using only hard
MaxSim?

## Why This Is Worth Testing

The previous best pure late-interaction result on this benchmark came from
query-side fusion alone:

- candidate policy: `query2doc_gpt4 + 1.9 * lamer_gpt4`
- exact full-index MaxSim quality: `nDCG@10 = 0.2936720442048378`

That policy already proved that query reformulation mattered more than the
baseline single-query MaxSim result. The remaining open question was whether
hard MaxSim itself was unnecessarily brittle for reasoning-heavy retrieval.

This was worth testing because recent late-interaction work suggests that the
MaxSim interaction matrix contains non-trivial redundancy and that the scoring
formulation is not sacred:

- `Col-Bandit` (arXiv:2602.02827, submitted February 2, 2026) reports that
  late-interaction MaxSim scoring contains enough redundancy to prune query-time
  computation while preserving ranking fidelity
- `ColBERT-Att` (arXiv:2603.25248, submitted March 26, 2026) explicitly argues
  that pure MaxSim ignores token-importance structure and proposes an
  attention-augmented late-interaction formulation

These papers do not evaluate the exact same operator used here, but they make
the broader design decision sound: it is reasonable to test alternatives to
hard MaxSim inside a pure late-interaction pipeline.

## Implementation Added

Added:

- [reference_topk_pooling.py](../../python/kayak_bridge/reference_topk_pooling.py)
  as an exact NumPy reference scorer for top-`k` token pooling
- [reference_softmax_pooling.py](../../python/kayak_bridge/reference_softmax_pooling.py)
  as an exact NumPy reference scorer for smooth temperature-weighted pooling
- [r2med_biology_late_interaction_rescoring.py](../../python/kayak_bridge/r2med_biology_late_interaction_rescoring.py)
  as a reproducible shortlist-rescoring benchmark that separates:
  - full-index candidate scoring
  - shortlist materialization
  - shortlist rescoring
- [bench_r2med_biology_late_interaction_rescoring.py](../../python/scripts/bench_r2med_biology_late_interaction_rescoring.py)
  as the script entrypoint

Why the benchmark is structured this way:

- candidate generation and shortlist materialization were separated from the
  operator timing on purpose, because otherwise materialization overhead would
  pollute claims about the operator itself
- this was especially important after observing that shortlist materialization
  was more expensive than rescoring for the winning operator

## Validation

Mechanical checks:

- [test_reference_topk_pooling.py](../../python/tests/test_reference_topk_pooling.py)
- [test_reference_softmax_pooling.py](../../python/tests/test_reference_softmax_pooling.py)
- [test_score_fusion.py](../../python/tests/test_score_fusion.py)
- [test_r2med_biology_query_variants.py](../../python/tests/test_r2med_biology_query_variants.py)
- [test_r2med_biology_late_interaction_rescoring.py](../../python/tests/test_r2med_biology_late_interaction_rescoring.py)
- [test_batch_api.py](../../python/tests/test_batch_api.py)
- [test_late_interaction.py](../../python/tests/test_late_interaction.py)

Benchmark command:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 python python/scripts/bench_r2med_biology_late_interaction_rescoring.py \
    --snapshot-root .cache/kayak/r2med_biology_full_main/docs_full_queries_full/store \
    --variant-cache-root .cache/kayak/r2med_biology_full_main/docs_full_queries_full/query_variants \
    --shortlist-k 100
```

Result artifact:

- [.cache/kayak/r2med_biology_late_interaction_rescoring_summary.json](../../.cache/kayak/r2med_biology_late_interaction_rescoring_summary.json)

## Measured Result

Full-index candidate policy:

- dataset: `R2MED/Biology`
- corpus: `57,359` documents
- queries: `103`
- retriever model: `colbert-ir/colbertv2.0`
- candidate policy: `query2doc_gpt4 + 1.9 * lamer_gpt4`
- shortlist size for the main operator sweep: `100`
- mean shortlist document-vector count: `11,567.864077669903`

Measured full-index candidate-score times:

- `query2doc_gpt4`: `28.115525041008368s`
- `lamer_gpt4`: `27.11138904094696s`

Measured shortlist materialization after the `LateIndex.select(...)`
optimization:

- total: `0.7916785000124946s`
- per query: `0.0076861990292475205s`

Measured operator results on that same shortlist:

- `maxsim_shortlist`:
  `nDCG@10 = 0.2936720442048378`,
  `MRR@10 = 0.37484974572353214`,
  `rescoring = 0.0023047228929252155s/query`
- `topk_mean_k2`:
  `nDCG@10 = 0.30548684254805375`,
  `MRR@10 = 0.40992063492063496`,
  `rescoring = 0.005559427184459654s/query`
- `topk_mean_k4`:
  `nDCG@10 = 0.310434232465269`,
  `MRR@10 = 0.40921944829711826`,
  `rescoring = 0.005810488261806878s/query`
- `topk_mean_k8`:
  `nDCG@10 = 0.3001940019916449`,
  `MRR@10 = 0.4007666820773618`,
  `rescoring = 0.0062249611647239005s/query`
- `softmax_tau0.25`:
  `nDCG@10 = 0.20046740888060968`
- `softmax_tau0.5`:
  `nDCG@10 = 0.16725369417198352`
- `softmax_tau1.0`:
  `nDCG@10 = 0.15099837433204422`

## Interpretation

Verified facts:

- replacing hard MaxSim with top-`k` mean pooling improves pure
  late-interaction quality on full `R2MED/Biology`
- the best measured operator in this sweep is `topk_mean_k4`
- the gain over hard MaxSim is:
  - absolute: `0.0167621882604312 nDCG@10`
  - relative: about `5.71%`
- smooth softmax pooling is not competitive in this setup and is a clear
  regression

Reasonable inference, not yet proven:

- `topk_mean_k4` likely helps because it rewards repeated token-level evidence
  instead of letting one single strongest token dominate every query vector
- that is consistent with a reasoning-heavy benchmark where many near-matching
  but contradictory or incomplete passages exist

This inference fits the data, but it is still an inference. Proving it would
require targeted per-query error analysis.

## Shortlist Threshold

An additional sweep tested the winning operator `topk_mean_k4` on smaller
candidate shortlists. Measured results:

- shortlist `10`:
  `nDCG@10 = 0.308248493850725`
- shortlist `20`:
  `nDCG@10 = 0.3072092225308967`
- shortlist `30`:
  `nDCG@10 = 0.310434232465269`
- shortlist `40`:
  `nDCG@10 = 0.310434232465269`
- shortlist `50`:
  `nDCG@10 = 0.310434232465269`
- shortlist `100`:
  `nDCG@10 = 0.310434232465269`
- shortlist `200`:
  `nDCG@10 = 0.310434232465269`

What this means:

- the `topk_mean_k4` gain saturates by shortlist `30`
- increasing the shortlist beyond `30` did not improve quality in this
  benchmark run
- that strongly suggests the quality gain comes from reranking already-strong
  candidates, not from needing a deeper candidate pool

## Materialization Bottleneck and Fix

Before optimizing [late_index.py](../../python/kayak_bridge/late_index.py),
shortlist materialization for `100` candidates measured:

- total: `1.3868742500199005s`
- per query: `0.01346479854388253s`

That was materially slower than the winning `topk_mean_k4` rescoring itself:

- `0.005857877028672007s/query`

This justified optimizing `LateIndex.select(...)` directly rather than blaming
the operator.

Implemented change:

- cache one doc-id to position map per `LateIndex`
- copy selected token ranges into one preallocated output matrix instead of
  building a list of matrices and concatenating them

Measured effect on the exact same shortlist-100 benchmark:

- materialization total improved from `1.3868742500199005s` to
  `0.7916785000124946s`
- per-query materialization improved from `0.01346479854388253s` to
  `0.0076861990292475205s`
- speedup: about `1.75x`

Quality stayed unchanged, which is what should happen for a pure
materialization optimization.

## External Comparison Boundary

Current public reference source:

- `https://r2med.github.io/`

Observed public Biology top on April 15, 2026:

- top public `Biology` score shown on the leaderboard: `54.01`

Comparison boundary:

- the Kayak run here is still pure late interaction with fused public query
  rewrites, not a direct copy of one leaderboard row
- so the comparison is informative but not row-for-row identical

Still, the broad conclusion is sound:

- best current pure late-interaction Kayak result on this setup:
  about `31.04` on the leaderboard-style `0-100` scale
- current public Biology frontier shown on the project page:
  `54.01`

So Kayak improved substantially over its own earlier pure late-interaction
baseline, but it is still clearly below the current public frontier on this
dataset.

## Conclusion

Verified:

- hard MaxSim is not the best pure late-interaction operator we have measured on
  `R2MED/Biology`
- `topk_mean_k4` is the current best measured pure late-interaction operator in
  this repo for this benchmark
- the gain is stable once the shortlist reaches `30`
- shortlist materialization was a real bottleneck and the `LateIndex.select(...)`
  optimization reduced it by about `1.75x`

Best next step:

- if we keep pushing this direction, the highest-leverage systems work is now
  candidate materialization and view-based candidate subsets, because operator
  math is no longer the dominant stage-2 cost
