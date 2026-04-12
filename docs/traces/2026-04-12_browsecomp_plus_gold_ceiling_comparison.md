# 2026-04-12 BrowseComp-Plus Gold Ceiling Comparison

## Goal

Close Phase I6 from
[docs/late_interaction_efficiency_roadmap.md](../late_interaction_efficiency_roadmap.md):

- implement one stronger local retrieval ceiling
- compare Kayak's current stage-aware path against it on the same queries
- label clearly what is exact, approximate, expensive, and unverified

## Files Added Or Changed

- `kayak/benchmarks/ceiling_comparison_json.mojo`
- `benchmarks/browsecomp_plus_gold_ceiling_comparison.mojo`
- `tests/test_ceiling_comparison_json.mojo`
- `pyproject.toml`

## Design Choice

Reason:
- the repo already had a clause-text reranker and an exploratory BrowseComp
  benchmark for it
- the missing piece was a machine-readable comparison against the current
  vector-only paths

Decision:
- treat the clause-text reranker as the current local stronger ceiling
- keep it explicitly labeled as:
  - `exact_full_scan` candidate generation
  - `clause_text` reranking

Important boundary:
- this is **not** a cross-encoder
- this is **not** a long-context LLM
- it is a richer local text-aware ceiling than exact MaxSim alone

## Verification Commands

```bash
pixi run test_ceiling_comparison_json
pixi run bench_browsecomp_plus_gold_ceiling_comparison_raw
```

Artifact:

```text
.cache/kayak/browsecomp_plus_gold_ceiling_comparison.json
```

## Measurement Context

Slice:

- `family = "browsecomp_plus"`
- `slice = "browsecomp_plus_gold_slice"`
- `final_k = 10`

Compared methods:

- exact full scan
- stage-aware `document_proxy` with `candidate_k = 20`
- stage-aware `document_proxy` with `candidate_k = 40`
- exact full scan plus clause-text rerank with `candidate_k = 20`
- exact full scan plus clause-text rerank with `candidate_k = 40`

## Result Snapshot

### Exact Baseline

- exact full scan
  - `candidate_k = 10`
  - `nDCG@10 = 0.2851`
  - `search = 0.001152 s`

### Stage-Aware Path

- `document_proxy`, `candidate_k = 20`
  - `candidate recall@10 = 0.95`
  - `nDCG@10 = 0.2123`
  - `search = 0.000947 s`
- `document_proxy`, `candidate_k = 40`
  - `candidate recall@10 = 1.0`
  - `nDCG@10 = 0.2851`
  - `search = 0.002129 s`

Interpretation:
- the current cheap faithful stage-aware point is still `document_proxy` at
  `candidate_k = 40`
- that point matches exact vector-only quality while staying faster than the
  local text-aware ceiling

### Stronger Local Ceiling

- exact full scan + clause-text rerank, `candidate_k = 20`
  - `candidate recall@10 = 1.0`
  - `nDCG@10 = 0.3638`
  - `search = 0.03442 s`
- exact full scan + clause-text rerank, `candidate_k = 40`
  - `candidate recall@10 = 1.0`
  - `nDCG@10 = 0.3233`
  - `search = 0.07903 s`

Interpretation:
- the local text-aware ceiling does beat exact MaxSim on this slice
- it is also much more expensive:
  - about `29.9x` slower than exact full scan at `candidate_k = 20`
  - about `37.1x` slower than `document_proxy` at `candidate_k = 40`

## What This Verifies

Verified locally:
- Kayak now has one explicit stronger ceiling benchmark
- that ceiling is richer than exact late interaction alone
- the benchmark compares it against the current stage-aware path on the same
  BrowseComp gold queries

## What This Still Does Not Prove

Not verified by this work:
- a cross-encoder ceiling
- a long-context LLM ceiling
- that clause-text reranking is the best possible expensive reference path

## Takeaway

Phase I6 now has a concrete local answer:
- there is real quality above exact MaxSim on this slice
- Kayak can measure that gap locally
- the current local ceiling is expensive enough that the latency-quality tradeoff
  is explicit, not rhetorical

That is enough to replace vague external ceiling language with one verified
local comparison, while still keeping the stronger long-context claims marked as
future work.
