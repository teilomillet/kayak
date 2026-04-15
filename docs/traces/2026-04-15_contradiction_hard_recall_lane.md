# 2026-04-15 Contradiction Hard-Recall Lane

## Goal

Add one harder synthetic benchmark family that directly stresses the failure
mode described in this session:

- same topic
- highly overlapping information
- opposite polarity
- only one polarity should count as relevant for the query

Reason:
- the existing `synthetic_hard_recall` lane is useful for conjunction pressure
- it does **not** directly model the more dangerous case where a retrieved
  document is topically correct but semantically contradictory
- legal, scientific, and medical retrieval all run into this polarity problem

## Sources Checked

Recent or canonical benchmark sources checked on `2026-04-15`:

- SciFact paper:
  - https://aclanthology.org/2020.emnlp-main.609.pdf
- HEALTHVER paper:
  - https://aclanthology.org/2021.findings-emnlp.297.pdf
- LegalBench-RAG:
  - https://arxiv.org/abs/2408.10343
- A Reasoning-Focused Legal Retrieval Benchmark:
  - https://law.stanford.edu/wp-content/uploads/2025/03/3709025.3712219.pdf

What those sources justify:

- SciFact established a retrieval-plus-verification setting where scientific
  evidence may support or refute a claim.
- HEALTHVER established the same basic structure for health-related claims and
  showed that evidence materially changes outcomes.
- LegalBench-RAG emphasizes precise retrieval of minimal relevant segments
  rather than broad topic matches.
- the 2025 Stanford legal benchmark argues that realistic legal retrieval often
  has low lexical overlap and requires more than shallow matching.

Inference:
- a Kayak benchmark ladder that wants to say something useful about
  contradiction-heavy retrieval should contain one explicit polarity-sensitive
  lane, not only conjunction and long-document lanes.

## Implementation

Added:

- [contradiction_hard_recall_fixture.mojo](../../kayak/benchmarks/contradiction_hard_recall_fixture.mojo)
- [contradiction_hard_recall_stage_aware.mojo](../../benchmarks/contradiction_hard_recall_stage_aware.mojo)
- [test_contradiction_hard_recall_fixture.mojo](../../tests/test_contradiction_hard_recall_fixture.mojo)

Minimal wiring:

- [__init__.mojo](../../kayak/benchmarks/__init__.mojo)
- [pyproject.toml](../../pyproject.toml)
- [benchmark_ladder.md](../benchmark_ladder.md)
- [hard_recall_evaluation.md](../hard_recall_evaluation.md)

Design choice:
- keep this lane compatible with the existing stage-aware benchmark machinery

Reason:
- the repo already has exact-reference candidate recall, stage densities, and
  plan-by-plan reporting
- reusing that surface keeps comparisons auditable
- adding a completely new evaluation path would weaken the epistemic quality of
  the result

## Fixture Shape

The new family uses:

- several topical slot values, like `synthetic_hard_recall`
- one explicit polarity vector
- matching-polarity documents
- opposite-polarity documents that share every topical slot
- one-slot near-miss adversaries for additional conjunction pressure

Current profile:

- `contradiction_hard_recall`
  - `slots6_values3_docs2988`

Key decision:
- use `primary_metric = "ndcg"`

Reason:
- `recall@k` alone is too weak for this lane
- a shortlist that includes contradictory documents in top positions should be
  penalized even if some relevant document is still retrieved later
- `nDCG` better reflects the ranking harm from contradictory near-duplicates

## Verification

Fixture tests:

```bash
pixi run test_contradiction_hard_recall_fixture
```

Observed:
- `4/4` tests passed

Benchmark run:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- \
  pixi run bench_contradiction_hard_recall_stage_aware_raw
```

Observed artifact:

- `.cache/kayak/contradiction_hard_recall_stage_aware_search.json`

Quiet-wrapper metadata:

- `.cache/kayak/bench_quiet/20260415T160229Z/meta.txt`

## Host-Noise Caveat

This was not a quiet-host run.

The wrapper timed out after `20s` and force-ran the benchmark because competing
CPU stayed around `287-302%` other CPU, with `Zed` and several
`modular-crashpad-handler` processes dominating the host.

Interpretation:
- the absolute timing numbers should be treated as host-contended evidence
- the relative recall and quality behavior across plans is still useful because
  all plans were measured in one bounded pass on the same host

## Results

Exact reference:

- `exact_full_scan`
  - `candidate_k = 2`
  - `candidate_recall = 1.0`
  - `primary = 1.0`
  - `mean_search_seconds ≈ 0.000688`

Approximate recovery frontier on the new contradiction lane:

| generator | `k=2` | `k=32` | `k=64` | `k=128` |
| --- | ---: | ---: | ---: | ---: |
| `document_proxy` | `0.0` | `0.25` | `1.0` | `1.0` |
| `centroid_postings` | `0.0` | `0.25` | `1.0` | `1.0` |
| `centroid_postings_imputed` | `0.0` | `0.25` | `1.0` | `1.0` |
| `centroid_heads` | `0.0` | `0.0` | `0.0833` | `0.0833` |

Numbers above are both:

- `mean_candidate_recall_at_final_k`
- `primary_value`

Reason:
- for this lane and current profile, when the oracle is missing from the
  shortlist the final ranking quality collapses with it

## Interpretation

This lane is behaving as intended.

Verified behavior:

- exact search is perfect
- small candidate budgets are catastrophically insufficient
- several approximate families need materially larger `candidate_k` than on the
  easier conjunction-style lane before they stop being dominated by
  opposite-polarity near-duplicates
- `centroid_heads` remains especially brittle on this profile

Why this is useful:

- it distinguishes "topic match" from "query-faithful match"
- it gives the repo a benchmark lane that better matches medical, scientific,
  and legal settings where contradictory evidence is common
- it lets Kayak optimize for a concrete failure mode instead of only saying
  benchmarks should be harder in the abstract

## Next Step

The next justified extension is not another easy public slice.

It is one of these:

1. Add a second contradiction profile with longer documents or more duplicates.
2. Mirror the same idea into a real-data lane such as SciFact-style evidence
   retrieval or a legal snippet benchmark.
3. Use this new lane as an optimization target for candidate generators that
   currently recover too late relative to exact search.
