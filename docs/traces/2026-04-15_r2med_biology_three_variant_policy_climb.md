# 2026-04-15 R2MED Biology Three-Variant Policy Climb

## Question

After verifying that `topk_mean_k4` beats hard MaxSim on the shortlist, can
Kayak climb further on full `R2MED/Biology` by adding one more public
query-rewrite variant while staying purely late-interaction end to end?

## Starting Point

Before this loop, the best verified pure late-interaction result on the full
`R2MED/Biology` benchmark was:

- candidate policy: `query2doc_gpt4 + 1.9 * lamer_gpt4`
- shortlist rescoring operator: `topk_mean_k4`
- shortlist size: `30+` was already shown to be enough
- quality:
  - `nDCG@10 = 0.310434232465269`
  - `MRR@10 = 0.40921944829711826`

This starting point was already documented in
[2026-04-15_r2med_biology_late_interaction_operator_sweep.md](./2026-04-15_r2med_biology_late_interaction_operator_sweep.md).

## Why Search This Direction

This next step was justified by local evidence, not guesswork:

- all public R2MED query-rewrite variants were already available through the
  existing cache path in
  [r2med_biology_query_variants.py](../../python/kayak_bridge/r2med_biology_query_variants.py)
- full exact score vectors can be fused cheaply once computed, so policy search
  is a reasonable loop on this benchmark
- the repo had no reproducible policy-expansion benchmark surface for this
  benchmark, so the search result would not be durable unless it was codified

## What Was Added

Added:

- [r2med_biology_late_interaction_policy_expansion.py](../../python/kayak_bridge/r2med_biology_late_interaction_policy_expansion.py)
  to run a reproducible seed-policy expansion sweep
- [bench_r2med_biology_late_interaction_policy_expansion.py](../../python/scripts/bench_r2med_biology_late_interaction_policy_expansion.py)
  as the command-line entrypoint

Updated:

- [r2med_biology_late_interaction_rescoring.py](../../python/kayak_bridge/r2med_biology_late_interaction_rescoring.py)
  so the default candidate policy now matches the best currently verified pure
  late-interaction rescored fusion
- [r2med_biology_late_interaction_quality.py](../../python/kayak_bridge/r2med_biology_late_interaction_quality.py)
  so the best currently verified three-variant maxsim-only policy is part of
  the default policy list

## Search Procedure

The search stayed bounded and explicit:

1. Keep the seed policy fixed at:
   - `query2doc_gpt4 = 1.0`
   - `lamer_gpt4 = 1.9`
2. Try each remaining public R2MED rewrite variant as one added third variant.
3. Sweep third-variant weights over:
   - `0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0, 1.25`
4. Measure full-index maxsim fusion quality for every expansion policy.
5. Identify the best added-variant family from that maxsim sweep.
6. Run the expensive shortlist-rescored sweep only for that single best
   added-variant family using `topk_mean_k4` and `shortlist_k = 30`.

This was chosen because:

- `shortlist_k = 30` had already been shown to saturate the `topk_mean_k4`
  gain on this benchmark
- the operator was already fixed by the previous loop, so the remaining search
  axis was policy choice rather than rescoring semantics
- rescoring every expansion policy was unnecessary once the maxsim sweep had
  already identified the strongest added-variant family

## Measured Results

Single-variant maxsim results remained:

- `lamer_gpt4`: `0.26924482104679437`
- `query2doc_gpt4`: `0.21430454589623862`
- best added non-seed single among the remaining variants was much lower

Best maxsim-only three-variant fusion found in the expansion sweep:

- policy:
  - `query2doc_gpt4 = 1.0`
  - `lamer_gpt4 = 1.9`
  - `search_r1_qwen7b_ins = 0.75`
- quality:
  - `nDCG@10 = 0.30262900650606983`
  - `MRR@10 = 0.3893435043920482`

Best shortlist-rescored three-variant fusion found in the same loop:

- policy:
  - `query2doc_gpt4 = 1.0`
  - `lamer_gpt4 = 1.9`
  - `search_r1_qwen7b_ins = 0.35`
- shortlist rescoring:
  - operator: `topk_mean_k4`
  - shortlist size: `30`
- quality:
  - `nDCG@10 = 0.32053144514255916`
  - `MRR@10 = 0.42432192941901686`
  - `recall@10 = 0.3587393614731377`

## Local Refinement Check

A follow-up local grid around the new three-variant policy checked whether
slightly changing the original seed weights could beat the simpler result.

The best local-grid alternatives did not improve on the simpler policy:

- best checked local alternative:
  - `query2doc_gpt4 = 1.1`
  - `lamer_gpt4 = 2.05`
  - `search_r1_qwen7b_ins = 0.4`
  - `nDCG@10 = 0.3203258484483581`
- best overall remained:
  - `query2doc_gpt4 = 1.0`
  - `lamer_gpt4 = 1.9`
  - `search_r1_qwen7b_ins = 0.35`
  - `nDCG@10 = 0.32053144514255916`

That means the current best policy is not only better, but also simpler.

## Interpretation

Verified facts:

- adding one more public query-rewrite variant can improve this benchmark while
  staying strictly late-interaction only
- the best additional variant found in this loop is
  `search_r1_qwen7b_ins`
- the best weight for that added variant under shortlist rescoring is `0.35`
- the best weight for that same added variant under maxsim-only fusion is
  different (`0.75`)

This matters because it shows:

- the best candidate-generation policy is not always the best final rescoring
  policy
- the shortlist rescoring operator and the candidate policy are coupled

## Quality Delta

Improvement over the previous best pure late-interaction result:

- previous best: `0.310434232465269`
- new best: `0.32053144514255916`
- absolute gain: `0.010097212677290174`
- relative gain: about `3.25%`

Improvement over the original full-index exact single-query baseline:

- original baseline: `0.09529319923295203`
- new best: `0.32053144514255916`
- absolute gain: `0.22523824590960713`

## Artifact and Command

Command:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 python python/scripts/bench_r2med_biology_late_interaction_policy_expansion.py \
    --snapshot-root .cache/kayak/r2med_biology_full_main/docs_full_queries_full/store \
    --variant-cache-root .cache/kayak/r2med_biology_full_main/docs_full_queries_full/query_variants \
    --shortlist-k 30 \
    --match-k 4
```

Primary artifact:

- `.cache/kayak/r2med_biology_late_interaction_policy_expansion_summary.json`

## Conclusion

The current best verified pure late-interaction `R2MED/Biology` result in this
repo is now:

- `query2doc_gpt4 = 1.0`
- `lamer_gpt4 = 1.9`
- `search_r1_qwen7b_ins = 0.35`
- shortlist rescoring with `topk_mean_k4`
- `nDCG@10 = 0.32053144514255916`

That is the policy now wired in as the default candidate policy for the
rescoring benchmark surface.
