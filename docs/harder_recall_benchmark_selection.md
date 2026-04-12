# Harder-Recall Benchmark Selection

Date: April 12, 2026

## Goal

Close Phase I5 from
[docs/late_interaction_efficiency_roadmap.md](late_interaction_efficiency_roadmap.md):

- state clearly whether the current public slices are enough
- select the next harder family if they are not
- record whether that family is implemented locally yet

## Sources Checked

- [docs/hard_recall_evaluation.md](hard_recall_evaluation.md)
- [docs/traces/2026-04-12_hard_recall_stage_aware.md](traces/2026-04-12_hard_recall_stage_aware.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md](traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md](traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md](traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md)
- [docs/traces/2026-04-12_synthetic_hard_recall_stage_aware.md](traces/2026-04-12_synthetic_hard_recall_stage_aware.md)

## Verified Facts

- `LIMIT-small` is still useful as a compact regression slice, but not as the
  strongest stage-1 stress test in this repo.
- `BrowseComp-Plus` gold is currently the strongest verified public hard-recall
  slice in the repo.
- that slice is still only `90` documents, so it is strong for query difficulty
  but weak for asymptotic candidate-generation claims.
- the local text-aware ceiling can outperform exact MaxSim on that slice, which
  means there is still headroom above the current vector-only path.
- the repo now has one implemented scalable synthetic conjunction-style family
  with explicit vector counts and exact-reference candidate recall reporting.

## Decision

The current small public slices are **not** the final benchmark bar for hard
stage-1 recall.

The next harder family should be:
- a larger retrieval-only BrowseComp-style corpus path, or
- a scalable synthetic conjunction-style family that is explicitly designed to
  stress stage-1 recall without hiding behind tiny corpora

## Selected Next Target

The explicit next benchmark target is:
- a scalable conjunction-style hard-recall synthetic family with explicit
  vector-count control and exact-reference recall reporting

Why this is the right next target:
- it can scale beyond the current `90`-document public slice
- it can separate query difficulty from corpus-size effects
- it can stress stage-1 recall directly instead of only final reranking quality

## Local Implementation

That selected target is now implemented locally.

Evidence:
- [docs/traces/2026-04-12_synthetic_hard_recall_stage_aware.md](traces/2026-04-12_synthetic_hard_recall_stage_aware.md)

Verified from that trace:
- the family now exists as runnable repo code
- it includes two measured profiles beyond the small public slices
- the tested approximate stage-1 plans require materially larger
  `candidate_k` than they do on the current `90`-document BrowseComp gold slice

## What This Verifies

Verified after this note:
- Kayak has an explicit benchmark-selection decision for the next harder family
- the repo is no longer pretending that the current small public slices are
  already sufficient on their own
- Phase I5 is now closed with an implemented synthetic hard-recall family

## What This Does Not Claim

This note does not claim:
- that the current public slices are useless
- that the current synthetic family replaces a larger public hard-recall corpus
- that the eventual harder family must be synthetic rather than public

It only records the honest state:
- the current public slices are informative
- they are not the last word
- the next harder benchmark family was selected explicitly and then implemented
  locally
