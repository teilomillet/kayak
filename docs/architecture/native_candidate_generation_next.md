# Native Candidate Generation Next Step

Status: `implemented baseline and measured follow-on`  
Date: `2026-04-12`

This note records what the new evidence in `kayak` does and does not justify.

Measured trace:
- [docs/traces/2026-04-12_centroid_postings_baseline.md](../traces/2026-04-12_centroid_postings_baseline.md)
- [docs/traces/2026-04-12_centroid_postings_flat.md](../traces/2026-04-12_centroid_postings_flat.md)

## Verified In Repo

These statements are now verified by the codebase and generated artifacts:

1. `kayak` has a real non-default stage-1 generator:
   - `exact_full_scan`
   - `document_proxy`
   - `centroid_postings`
2. Stage 2 is now a real exact late-interaction rerank over the shortlisted
   documents, not just a truncation of stage-1 hits.
3. Sealed segments can now persist search-native sidecars:
   - `document_proxy/`
   - `centroid_postings/`
4. The public artifact layer now exposes:
   - candidate-window recall by generator family
   - vector-budget sweeps for both non-default generators

## What The New Evidence Says

The `document_proxy` path is useful because it gives a cheap baseline for:
- candidate recall versus `candidate_k`
- sensitivity to query vector budgets
- sensitivity to stage-1 document vector budgets

It is not yet the final engine direction for `kayak`.

Reason:
- it still uses one proxy vector per document
- it does not preserve the richer local structure that makes late interaction
  attractive in the first place
- it is a baseline proxy family, not a native multi-vector index

The `centroid_postings` path is useful because it verifies a second point in the
design space:
- the sidecar can retain more document-local structure than one proxy vector
  per document
- the planner and artifact layers can compare it against `document_proxy` and
  `exact_full_scan` on identical slices

What the current public traces say:
- `centroid_postings` is now measurable on all public slices
- on the current slices, it is usually weaker than `document_proxy` at the same
  `candidate_k`
- it still reaches full oracle recall once `candidate_k` approaches the full
  corpus window

Inference:
- `centroid_postings` is a useful native baseline, but not yet evidence that a
  centroid/posting family automatically beats the lighter proxy baseline

## Measured Follow-On

The first tighter native-layout follow-on is now also measured:

- `centroid_postings_flat`

What the current public traces say:

- it preserves the same quality as `centroid_postings` on the public
  vector-budget sweep
- it preserves the same candidate recall and judged quality on the hard-recall
  public slice
- it slightly improves average stage-1 / search latency on the current public
  artifacts
- its per-slice stage-1 profile is still mixed, so it is not yet a justified
  silent replacement everywhere

Inference:

- some native-engine improvement is available already from layout tightening
  alone
- this is stronger evidence than the earlier blockmax step because it improves
  the implementation shape without paying a faithfulness penalty on the current
  public artifacts

## Sound Next Native Step

The next heavier engine step should now build on the tighter native layout, not
on another heuristic sidecar layered over the older posting scan.

The best next candidate is:
- a WARP or GEM-style native multi-vector engine path built on top of:
  - the verified stage boundary
  - the sealed sidecar benchmarks
  - the now-measured flat centroid execution layout

Why this is the sound next step:
- we already have:
  - one light proxy baseline
  - one more native centroid/posting baseline
  - one tighter semantics-preserving native follow-on
  - exact stage-2 reranking
  - oracle-recall and vector-budget traces
- the remaining uncertainty is now about engine-native layout and pruning
  quality, not about whether `kayak` can support multiple stage-1 families

## What We Should Not Do

We should not keep adding many more heuristic sidecars before native-engine work.

That would risk:
- spending cycles on baseline families that are already sufficient to expose the
  current recall tradeoffs
- delaying the real engine question, which is whether a tighter native design
  can beat both current baselines on the same traces

## Immediate Follow-On

The next implementation milestone should be:
- keep `document_proxy` and `centroid_postings` as regression baselines
- add one tighter native candidate engine inspired by WARP or GEM
- judge it on the same candidate-window, vector-budget, and storage traces
- only promote it if it improves the current Pareto frontier on those artifacts
