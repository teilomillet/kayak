# Centroid Head Generator

Date: `2026-04-12`

This step adds a more aggressive native centroid candidate generator without
claiming a new paper-derived engine.

## What Changed

The centroid-postings sidecar now has an explicit posting-order contract:
- postings are stored in `weight_desc_doc_asc` order within each centroid
- the order is persisted as `posting_order_kind` in the centroid sidecar
  manifest

On top of that layout, `kayak` now exposes:
- `centroid_postings_head`

Its rule is simple and explicit:
- pick the same top centroid probes as the baseline centroid stage
- scan only the first `min(candidate_k, 16)` postings from each selected
  centroid
- keep exact late interaction as stage 2 over the shortlisted documents

This is intentionally not presented as WARP, PLAID, or GEM.
It is a small native layout-aware baseline that uses the sorted centroid
postings already stored by `kayak`.

## Guardrails

Verified:
- old centroid sidecars without `posting_order_kind` still load
- those old sidecars are marked `unspecified`
- `ensure_stored_centroid_posting_index()` upgrades them to the new sorted
  layout
- `centroid_postings_head` refuses to run on unordered sidecars

That last guardrail matters because the head-truncation logic only has meaning
when the sidecar order is explicit.

## Candidate-Window Results

Generated artifact:
- `.cache/kayak/public_candidate_window_sweep.json`

Selected rows:

- `beir/scifact/test`, `candidate_k = 40`
  - `centroid_postings`: recall `0.95`, candidate stage `0.0000937s`
  - `centroid_postings_head`: recall `0.9`, candidate stage `0.0000935s`
- `orionweller/LIMIT-small`, `candidate_k = 20`
  - `centroid_postings`: recall `0.515625`, candidate stage `0.0000871s`
  - `centroid_postings_head`: recall `0.5125`, candidate stage `0.0000830s`
- `Tevatron/browsecomp-plus/evidence-slice`, `candidate_k = 20`
  - `centroid_postings`: recall `0.625`, candidate stage `0.0001028s`
  - `centroid_postings_head`: recall `0.625`, candidate stage `0.0000993s`
- `Tevatron/browsecomp-plus/evidence-slice`, `candidate_k = 40`
  - `centroid_postings`: recall `0.8`, candidate stage `0.0001008s`
  - `centroid_postings_head`: recall `0.825`, candidate stage `0.0001055s`

Inference:
- the head generator is not uniformly faster or better
- it is sometimes slightly faster for the same recall
- it is sometimes slightly more recall-friendly on a harder slice
- it is also sometimes worse than the simpler centroid baseline

## Vector-Budget Results

Generated artifact:
- `.cache/kayak/public_vector_budget_sweep.json`

Best candidate-recall rows by dataset:

- `beir/scifact/test`
  - `centroid_postings`: recall `0.95` at query budget `32`, document budget `128`
  - `centroid_postings_head`: recall `0.916667` at query budget `16`, document budget `128`
- `beir/fiqa/test`
  - `centroid_postings`: recall `0.9` at query budget `16`, document budget `64`
  - `centroid_postings_head`: recall `0.883333` at query budget `32`, document budget `128`
- `orionweller/LIMIT-small`
  - `centroid_postings`: recall `0.915625` at query budget `4`, document budget `8`
  - `centroid_postings_head`: recall `0.91875` at query budget `16`, document budget `128`
- `Tevatron/browsecomp-plus/evidence-slice`
  - `centroid_postings`: recall `0.9` at query budget `8`, document budget `64`
  - `centroid_postings_head`: recall `0.85` at query budget `32`, document budget `64`

Inference:
- `centroid_postings_head` is a legitimate additional Pareto point on some
  slices
- it does not dominate `centroid_postings`
- it should remain an explicit benchmarked option, not a silent replacement

## Decision

The sound conclusion is:
- keep `centroid_postings_head` as a non-default native baseline
- keep the weight-sorted posting order as the centroid sidecar default
- do not replace `centroid_postings` with it
- use all four current public stage-1 references when comparing native choices:
  - `document_proxy`
  - `centroid_postings`
  - `centroid_postings_head`
  - `centroid_postings_imputed`
