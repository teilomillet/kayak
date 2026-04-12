# Centroid Postings Baseline

Date: `2026-04-12`

This note records the first measured `centroid_postings` baseline in `kayak`.

Generated artifacts:
- `.cache/kayak/public_candidate_window_sweep.json`
- `.cache/kayak/public_vector_budget_sweep.json`
- `.cache/kayak/public_real_slice_collection_storage.json`

## Verified

- `centroid_postings` is now a real stage-1 candidate generator in the public
  benchmark artifacts.
- The stage-2 path remains exact late interaction over the shortlisted
  documents.
- The collection mirror now persists both:
  - `document_proxy/`
  - `centroid_postings/`

## Candidate-Window Recall

Selected values from `public_candidate_window_sweep.json`:

- `SciFact`, `candidate_k=40`
  - `document_proxy`: `1.0`
  - `centroid_postings`: `0.95`
- `FIQA`, `candidate_k=40`
  - `document_proxy`: `0.9333`
  - `centroid_postings`: `0.8833`
- `LIMIT-small`, `candidate_k=40`
  - `document_proxy`: `0.9656`
  - `centroid_postings`: `0.9031`
- `BrowseComp-Plus` evidence slice, `candidate_k=40`
  - `document_proxy`: `1.0`
  - `centroid_postings`: `0.8`

Inference:
- the current `centroid_postings` implementation is a valid native baseline,
  but it is not yet the best point on the current stage-1 recall frontier

## Vector-Budget Sweep

`public_vector_budget_sweep.json` now contains `100` `centroid_postings` rows
across the five public slices.

Selected best candidate-recall rows by dataset:

- `SciFact`
  - `document_proxy`: recall `1.0` at query budget `32`, document budget `64`
  - `centroid_postings`: recall `0.95` at query budget `32`, document budget `128`
- `BrowseComp-Plus` evidence slice
  - `document_proxy`: recall `1.0` at query budget `4`, document budget `64`
  - `centroid_postings`: recall `0.9` at query budget `8`, document budget `64`

Inference:
- the benchmark plumbing is now strong enough to compare multiple non-default
  stage-1 families on the same query-budget and document-budget axes

## Storage Cost

`public_real_slice_collection_storage.json` now includes the storage impact of
both sidecars.

Current bytes per vector:

- `SciFact`: `523.13`
- `FIQA`: `524.09`
- `LIMIT-small`: `524.45`
- `BrowseComp-Plus` evidence slice: `520.35`
- `BrowseComp-Plus` gold slice: `520.35`

Inference:
- adding the centroid sidecar increased bytes per vector relative to the prior
  document-proxy-only snapshot, so heavier native work should be judged on
  storage as well as recall and latency
