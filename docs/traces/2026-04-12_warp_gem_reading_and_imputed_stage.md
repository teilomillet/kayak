# WARP and GEM Reading, Then Imputed Centroid Stage

Date: `2026-04-12`

This note records the primary-source review that informed the next native step
after the initial `centroid_postings` baseline.

## Sources Checked

Primary sources:
- WARP paper: https://arxiv.org/abs/2501.17788
- WARP public code: https://github.com/jlscheerer/xtr-warp
- GEM paper: https://arxiv.org/abs/2603.20336

What was directly verified:
- WARP is still a centroid/residual engine, not a generic ANN wrapper.
- The WARP search loop in public code is:
  - query-centroid scoring
  - centroid selection with WARPSELECT
  - decompression of selected cells
  - token-level max reduction
  - document-level sum reduction with missing-similarity estimates
- WARP paper Section 4.3 describes missing similarity imputation using the first
  centroid score whose cumulative cluster size exceeds `t'`.
- WARP paper Section 4.5 describes the two-stage reduction:
  - token-level max across probed cells
  - document-level sum with imputed missing token scores
- GEM paper describes a native graph-based set-level index with:
  - set-level clustering
  - local graphs plus bridge structure
  - multi-entry beam search
  - early pruning using cluster cues

## Code Availability Boundary

Verified:
- public WARP code exists and was inspected locally from
  `jlscheerer/xtr-warp`

Inference:
- no public GEM implementation with enough evidence to reuse was found in the
  sources checked above on `2026-04-12`
- therefore the next sound implementation step in `kayak` is WARP-inspired,
  not GEM-inspired graph search

## Implemented Step

Implemented in this repo:
- a new `centroid_postings_imputed` stage-1 generator

This step intentionally copies only the parts justified by the current `kayak`
storage shape:
- size-aware centroid selection
- missing-similarity imputation
- token-level max merge across selected centroids
- document-level sum merge with imputed missing token scores

This step does **not** claim to reproduce full WARP:
- no residual compression
- no implicit decompression over quantized residuals
- no fused C++ kernels
- no XTR-specific compressed index layout

## Measured Outcome

Generated artifacts:
- `.cache/kayak/public_candidate_window_sweep.json`
- `.cache/kayak/public_vector_budget_sweep.json`

Candidate-window recall highlights:

- `SciFact`, `candidate_k=40`
  - `centroid_postings`: `0.95`
  - `centroid_postings_imputed`: `0.90`
- `FIQA`, `candidate_k=40`
  - `centroid_postings`: `0.8833`
  - `centroid_postings_imputed`: `0.8833`
- `LIMIT-small`, `candidate_k=40`
  - `centroid_postings`: `0.9031`
  - `centroid_postings_imputed`: `0.8688`
- `BrowseComp-Plus` evidence slice, `candidate_k=40`
  - `centroid_postings`: `0.8`
  - `centroid_postings_imputed`: `0.75`

Vector-budget best rows show a more mixed story:

- `BrowseComp-Plus` gold slice
  - `centroid_postings`: recall `0.9`, `nDCG@10 = 0.3059`
  - `centroid_postings_imputed`: recall `0.85`, `nDCG@10 = 0.35`
- `SciFact`
  - `centroid_postings`: recall `0.95`, `nDCG@10 = 0.7584`
  - `centroid_postings_imputed`: recall `0.9`, `nDCG@10 = 0.7718`

Inference:
- the imputed scorer is closer to WARP’s reduction logic
- on current public slices, it usually lowers stage-1 oracle recall relative to
  the simpler centroid baseline
- it can still improve downstream `nDCG` or `MRR` on some slices

## Decision

The sound conclusion is:
- keep `centroid_postings_imputed` as an explicit benchmarked engine option
- do not silently replace `centroid_postings` with it
- use the three current non-exact baselines as references:
  - `document_proxy`
  - `centroid_postings`
  - `centroid_postings_imputed`
- reserve the next heavier native step for layout changes that move the true
  Pareto frontier, not just the reduction heuristic
