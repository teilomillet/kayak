# Late-Interaction Efficiency Roadmap

This note records the roadmap implied by the stronger late-interaction thesis:

- single-core late interaction should scale much farther than people assume
- token representations should compress much harder than document identities
- stage-1 recall pressure should be measured on harder tasks
- the bar should not be only "beat dense retrieval on easy first-stage tasks"

This is intentionally epistemic.

The goal is to separate:
- what Kayak has already verified as engine substrate
- what Kayak has instrumented but not yet proved
- what remains a research claim that still needs numbers

## Sources Checked

- [TODO.md](../TODO.md)
- [docs/late_interaction_2030.md](late_interaction_2030.md)
- [docs/hard_recall_evaluation.md](hard_recall_evaluation.md)
- [docs/traces/2026-04-12_limit_browsecomp_public_slices.md](traces/2026-04-12_limit_browsecomp_public_slices.md)
- user-provided Omar Khattab workshop transcript discussed in repo on
  `2026-04-12`

## Status Key

- `Verified substrate`
  - the repo already has the interface or benchmark machinery needed
- `Instrumented but unproven`
  - the repo can measure the claim, but has not yet established it
- `Unverified claim`
  - the repo does not yet have sufficient local evidence
- `Deferred`
  - valuable, but not the next milestone

## Current Position

### Verified Substrate

- explicit `SearchPlan` stages with exact stage 2 reranking
- candidate recall against an exact full-scan reference
- machine-readable stage-aware benchmark output
- hosted collection create, mutate, snapshot, export/import, search, explain
- storage byte accounting and vector-count accounting
- public hard-recall slices for `BrowseComp-Plus` evidence and gold

This matters because these are the prerequisites for honest efficiency work.

### Instrumented But Not Yet Proved

- native non-exact stage-1 generators exist
- hard-recall benchmark output now distinguishes candidate recall from judged
  retrieval quality
- storage reports can measure bytes per document and bytes per vector

Inference:
- Kayak can now test the stronger thesis
- Kayak has not yet established that thesis

### Unverified Claims

These are the important unresolved claims from the stronger roadmap:

- under-`200ms` single-core search at multi-billion-token scale
- roughly `6 bytes/vector` storage without unacceptable quality loss
- document-vector pruning laws closer to `sqrt(m)` than today's typical
  heuristic budgets
- asymptotic latency substantially better than naive late interaction on large
  corpora
- hard-recall benchmark families materially beyond current `LIMIT-small` and
  current small BrowseComp slices
- comparisons against a much more expensive retrieval ceiling that are locally
  implemented and verified

## Parallel Tracks

Kayak should now run two tracks in parallel.

### Track A: Engine Continuity

Status:
- `Verified substrate`

Goal:
- keep the engine operable and inspectable while research work proceeds

This includes:
- storage/runtime cleanup
- service boundary work
- profiling surfaces
- safe native candidate-engine iterations

### Track B: Efficiency And Scaling Thesis

Status:
- `Instrumented but unproven`

Goal:
- convert the stronger late-interaction claims into local evidence or local
  falsification

This is the new primary research track.

## Research Phases

### Phase I1: Single-Core Scale Benchmark

Status:
- `Implemented and locally measured on a synthetic scale slice`

Claim:
- late interaction can search much larger corpora on one CPU core than common
  intuition suggests

Evidence needed:
- one reproducible benchmark that reports latency against document count, token
  count, vector count, and candidate budget on one core

Deliverables:
- a benchmark entrypoint for increasing corpus scale
- pinned reporting for query vector count, token count, bytes, and stage
  policy
- at least one trace note with measured results

Exit criteria:
- Kayak can state a verified single-core scaling curve instead of a qualitative
  claim

Current evidence:
- [docs/traces/2026-04-12_single_core_scale.md](traces/2026-04-12_single_core_scale.md)
- verified locally for a synthetic fixed-shape slice over `64 -> 4096`
  documents
- not yet evidence for multi-billion-token or clean idle-host claims

### Phase I2: Bytes-Per-Vector Compression

Status:
- `Unverified claim`

Claim:
- current late-interaction storage is materially over-provisioned, and a much
  smaller bytes-per-vector target is reachable

Evidence needed:
- measured bytes/vector before and after one explicit compressed token format
- correctness checks against exact scoring on representative queries
- retrieval-quality measurements on at least one public hard-recall slice

Deliverables:
- one non-default compressed token storage path
- one benchmark that reports:
  - bytes/vector
  - build time
  - search latency
  - retrieval quality

Exit criteria:
- Kayak can quantify the storage-quality-latency tradeoff of one compressed
  path

### Phase I3: Token Redundancy And Vector-Count Laws

Status:
- `Unverified claim`

Claim:
- contextualized tokens are redundant enough that aggressive vectors/document
  reduction is plausible without immediate collapse

Evidence needed:
- sweeps over retained vectors/document
- explicit reporting of stage-1 recall, final retrieval quality, latency, and
  bytes/vector

Deliverables:
- one benchmark family that varies vectors/document directly
- one note stating whether a `sqrt(m)`-style law is supported, unsupported, or
  still ambiguous

Exit criteria:
- Kayak can say something measured about vector-count laws rather than gesture
  at them

### Phase I4: Native Candidate Engine Asymptotics

Status:
- `Instrumented and partially measured on the synthetic scale slice`

Claim:
- native stage-1 late-interaction engines can scale asymptotically better than
  naive full-corpus scoring in ways that matter at larger corpus sizes

Evidence needed:
- scaling comparisons between exact full scan and native candidate engines over
  increasing corpus sizes
- stage-1 recall and final quality reported on the same runs

Deliverables:
- one benchmark suite that grows collection size while holding query workload
  fixed
- one comparison across at least:
  - exact full scan
  - `document_proxy`
  - one tighter native engine family

Exit criteria:
- Kayak can show where each candidate engine wins or loses on the
  latency-recall-quality frontier

Current evidence:
- the synthetic single-core scale benchmark already compares:
  - exact full scan
  - `document_proxy`
  - `centroid_postings`
  - `centroid_heads`
  - `centroid_postings_head_auto`
- it already exposed a real tradeoff:
  - capped native engines stayed fast
  - capped native engines also lost candidate recall as corpus size increased
- broader public-benchmark confirmation is still pending

### Phase I5: Harder-Recall Benchmark Selection

Status:
- `Instrumented but unproven`

Claim:
- current public slices are useful, but they are not yet the strongest stress
  tests for stage-1 recall

Evidence needed:
- one benchmark-selection note that justifies a harder family than the current
  default public slices
- either a public slice or a clearly documented synthetic generator

Deliverables:
- one new benchmark-selection note
- one new benchmark entrypoint or one justified deferral note

Exit criteria:
- Kayak has at least one harder-recall benchmark family beyond the current
  small public slices, or an explicit reason it does not yet

### Phase I6: Stronger Ceiling Comparisons

Status:
- `Unverified claim`

Claim:
- the real bar is not only dense retrieval; it is a much more expensive
  retrieval ceiling

Evidence needed:
- one locally implemented expensive reference path
- one benchmark that compares Kayak's stage-aware path against that ceiling on
  the same queries

Deliverables:
- a verified reference path
- one comparison note that labels clearly what is exact, approximate,
  expensive, and unverified

Exit criteria:
- Kayak can compare itself against a stronger ceiling without relying on vague
  external claims

## Immediate Parallel TODOs

These are the next moves that should happen while ongoing engine work
continues.

- [x] add one single-core scaling benchmark over increasing corpus sizes
- [ ] add one compressed-token benchmark that reports bytes/vector explicitly
- [ ] add one vectors/document sweep that tests aggressive document-vector
  reduction
- [x] add one asymptotic scaling benchmark for native candidate engines
- [ ] add one benchmark-selection note for a harder-recall family beyond the
  current default public slices
- [ ] add one stronger-ceiling comparison only after that path exists locally

## What Not To Claim Yet

- Do not claim the Omar-style single-core efficiency story without local
  scaling numbers.
- Do not claim `6 bytes/vector` viability without a measured compressed format.
- Do not claim `sqrt(m)` pruning laws without an explicit vectors/document
  sweep.
- Do not treat judged retrieval gains as equivalent to candidate recall against
  exact stage 2.
- Do not present current public hard-recall slices as the final benchmark bar.

## Decision Rule

When choosing the next roadmap item, prefer the one that:
- turns a qualitative efficiency claim into a measurable benchmark
- keeps vector counts, bytes, and stage policies explicit
- compares quality, recall, latency, and storage in the same artifact
- can falsify an attractive hypothesis, not only confirm it

Avoid work that:
- improves only one easy benchmark slice
- adds another heuristic without a scaling study
- hides stage semantics behind generic vector-DB language
- substitutes rhetoric for measurements
