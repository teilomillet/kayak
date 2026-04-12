# Hard-Recall Evaluation

This note defines the first hard-recall evaluation surface for Kayak's explicit
stage-1 plus stage-2 search plans.

It is intentionally narrower than a field-level benchmark manifesto.

The goal is:
- choose the slices that actually stress stage-1 recall
- define what Kayak's stage-aware benchmark should report
- separate repo-verified comparisons from open future comparisons

## Sources Checked

- [TODO.md](../TODO.md)
- [docs/late_interaction_2030.md](late_interaction_2030.md)
- [docs/traces/2026-04-12_limit_browsecomp_public_slices.md](traces/2026-04-12_limit_browsecomp_public_slices.md)

Verified from repo evidence:
- `LIMIT-small` is almost solved on the current exact ColBERTv2 slice.
- `BrowseComp-Plus` evidence and gold slices remain materially harder.
- the current repo can measure stage-1 candidate recall against an exact
  full-scan reference through `CollectionSearchExplain`.

## Selected Hard-Recall Slices

The current hard-recall public set is:
- `BrowseComp-Plus` evidence slice
- `BrowseComp-Plus` gold slice

Reason:
- both slices are already present in the repo
- both slices are harder than the current `LIMIT-small` public slice
- both slices let Kayak test stage-1 recall pressure without inventing a new
  benchmark distribution first

## Explicit Exclusion

`LIMIT-small` is excluded from this hard-recall set for now.

Reason:
- repo measurements show it is useful as a compact regression slice
- repo measurements do not show it as the strongest stage-1 recall stress test

This is a scope decision, not a claim that `LIMIT-small` is unimportant.

## What The Stage-Aware Benchmark Reports

Each summary reports:
- candidate generator kind
- faithfulness policy kind
- final `k`
- candidate `k`
- mean candidate recall at final `k`, measured against an exact full-scan
  reference
- final retrieval quality: primary metric, `nDCG@k`, `MRR@k`, `recall@k`,
  success rate
- mean search latency for the non-debug search path
- storage context: document count, token count, vector count, byte size,
  bytes per document, bytes per vector

That is the minimum shape needed to reason about the stage tradeoff honestly.

## What The Current Benchmark Does Not Claim

The current stage-aware benchmark still does not compare against a long-context
LLM or a cross-attention ceiling.

Reason:
- that comparison is not yet implemented or locally verified in this repo
- using an unverified external ceiling would weaken the epistemic quality of
  the benchmark

The repo now does have one stronger local comparison path:
- exact full scan plus `clause_text` reranking on BrowseComp gold
- recorded separately in
  [docs/traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md](traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md)

For the stage-aware benchmark itself, the verified reference ceiling remains:
- exact full-scan late interaction on the same collection snapshot

## Current Benchmark Entry Point

The current benchmark command is:

```bash
pixi run bench_hard_recall_real_subset
```

It writes:

```text
.cache/kayak/hard_recall_stage_aware_search.json
```

The current plan families in that artifact are:
- `exact_full_scan`
- `document_proxy`
- `centroid_heads`
- `centroid_postings`
- `centroid_postings_head`
- `centroid_postings_imputed`

The non-exact plans currently run under:
- `faithfulness_policy_kind = "best_effort"`

This output is the current source of truth for:
- how much stage-1 pruning reduces exact-reference recall
- how much final quality is recovered after exact rescoring
- what latency and storage context those comparisons were measured under
