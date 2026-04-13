## Goal

Take a step back from the implementation loop and encode the parts of Omar
Khattab's "Late Interaction in 2030" talk and Q&A that were still underrepresented
in the repo roadmap.

## Why this step

The repository already captured the core thesis:

- late interaction should be treated as a paradigm
- harder stage-1 recall pressure matters
- hosted-engine continuity matters

What was still too implicit in the roadmap:

- the benchmark ladder should stay explicit rather than growing ad hoc
- stronger ceilings must be labeled honestly
- code or beyond-text retrieval should be a later evaluation lane
- heterogeneous encoder late interaction is not something Kayak should imply is
  supported by default
- hosted deployment claims need measured numbers, not just algorithmic
  intuition

## Changes

Updated:

- `docs/late_interaction_2030.md`
- `TODO.md`

The updates add:

- a new "What The Q&A Adds" section
- explicit treatment of:
  - multimodal/tabular/code retrieval
  - generative retrieval as adjacent, not default
  - model-agnostic late interaction as unsupported without training/calibration
  - pretraining-for-late-interaction as an open question
  - deployment numbers as a first-class requirement
  - feature-complete hosted search-engine needs
- two new roadmap priorities:
  - harder task ladder and stronger ceilings
  - encoder boundary and interoperability

## Outcome

The repo roadmap is now closer to the actual field argument in the talk:

- not just "build a faster candidate engine"
- but also:
  - keep the benchmark ladder explicit
  - keep stronger ceilings honest
  - keep hosted infra first-class
  - keep interoperability claims epistemically narrow until proven

No code behavior changed in this step; this was a strategic roadmap correction.
