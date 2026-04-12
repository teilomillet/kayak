# Native Candidate Generation Next Step

Status: `informed next step`  
Date: `2026-04-12`

This note records what the new evidence in `kayak` does and does not justify.

## Verified In Repo

These statements are now verified by the codebase and generated artifacts:

1. `kayak` has a real non-default stage-1 generator:
   - `exact_full_scan`
   - `document_proxy`
2. Stage 2 is now a real exact late-interaction rerank over the shortlisted
   documents, not just a truncation of stage-1 hits.
3. Sealed segments can now persist one search-native sidecar:
   - `document_proxy/`
4. The public artifact layer now exposes:
   - candidate-window recall by generator family
   - vector-budget sweeps for the proxy stage

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

## Sound Next Native Step

The next heavier engine step should be a segment-native candidate family that
retains more local structure than one proxy vector per document.

The best next candidate is:
- a centroid or posting-based sidecar per sealed segment

Why this is the sound next step:
- it fits the current `SearchPlan` and segment-sidecar architecture
- it is closer to PLAID and MUVERA-style candidate generation than a single
  proxy vector baseline
- it preserves exact late interaction as the correctness anchor in stage 2
- it can be judged with the new oracle-recall and vector-budget artifacts

## What We Should Not Do

We should not jump directly from `document_proxy` to a large opaque native
engine rewrite without intermediate evidence.

That would risk:
- losing the exact correctness anchor
- mixing storage, planner, and candidate-generation changes in one step
- making it harder to attribute wins to the actual design change

## Immediate Follow-On

The next implementation milestone should be:
- add one segment-native centroid/posting sidecar
- compare it against `document_proxy` on the same candidate-window and
  vector-budget public slices
- use those traces to decide whether a heavier WARP/GEM-style native index is
  justified for the next cycle
