# Harder-Recall Benchmark Selection

Date: April 12, 2026

## Goal

Close Phase I5 from
[docs/late_interaction_efficiency_roadmap.md](late_interaction_efficiency_roadmap.md):

- state clearly whether the current public slices are enough
- select the next harder family if they are not
- record an explicit reason when that family is not yet implemented locally

## Sources Checked

- [docs/hard_recall_evaluation.md](hard_recall_evaluation.md)
- [docs/traces/2026-04-12_hard_recall_stage_aware.md](traces/2026-04-12_hard_recall_stage_aware.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md](traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md](traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md](traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md)

## Verified Facts

- `LIMIT-small` is still useful as a compact regression slice, but not as the
  strongest stage-1 stress test in this repo.
- `BrowseComp-Plus` gold is currently the strongest verified public hard-recall
  slice in the repo.
- that slice is still only `90` documents, so it is strong for query difficulty
  but weak for asymptotic candidate-generation claims.
- the local text-aware ceiling can outperform exact MaxSim on that slice, which
  means there is still headroom above the current vector-only path.

## Decision

The current small public slices are **not** the final benchmark bar for hard
stage-1 recall.

The next harder family should be:
- a larger retrieval-only BrowseComp-style corpus path, or
- a scalable synthetic conjunction-style family that is explicitly designed to
  stress stage-1 recall without hiding behind tiny corpora

## Why There Is No New Local Family Yet

This phase closes as a justified deferral rather than a new benchmark artifact.

Reason:
- a credible larger public family should not be smuggled in as an underspecified
  mini-slice
- a credible synthetic family should be designed intentionally, not improvised
  after the fact just to satisfy a checklist
- the current repo now has enough frontier evidence to justify the next family,
  but not enough to claim that it has already integrated one honestly

## Selected Next Target

The explicit next benchmark target is:
- a scalable conjunction-style hard-recall synthetic family with explicit
  vector-count control and exact-reference recall reporting

Why this is the right next target:
- it can scale beyond the current `90`-document public slice
- it can separate query difficulty from corpus-size effects
- it can stress stage-1 recall directly instead of only final reranking quality

## What This Verifies

Verified after this note:
- Kayak has an explicit benchmark-selection decision for the next harder family
- the repo is no longer pretending that the current small public slices are
  already sufficient
- Phase I5 is closed as a documented deferral with a concrete next family

## What This Does Not Claim

This note does not claim:
- that the next family is implemented yet
- that the current public slices are useless
- that the eventual harder family must be synthetic rather than public

It only records the honest state:
- the current public slices are informative
- they are not the last word
- the next harder benchmark family is now selected explicitly
