# Extensibility Wall For New Late-Interaction Work

Status: `architecture note`  
Date: `2026-04-13`

This note defines the "wall" that new late-interaction work should hit when it
enters `kayak`.

The goal is simple:

- new papers and implementations should be classified before they are coded
- most improvements should plug into an existing seam
- only a smaller class of work should require engine refactors

This is intentionally epistemic:

- verified repo facts are tied to the current codebase
- source-backed claims are tied to the referenced papers or public code/docs
- inferences are labeled as such

## Sources Checked

### Repository files

- [kayak/planning/candidate_generator.mojo](../../kayak/planning/candidate_generator.mojo)
- [kayak/planning/stage1_capabilities.mojo](../../kayak/planning/stage1_capabilities.mojo)
- [kayak/planning/execution.mojo](../../kayak/planning/execution.mojo)
- [kayak/planning/search_plan.mojo](../../kayak/planning/search_plan.mojo)
- [kayak/collections/search_artifact.mojo](../../kayak/collections/search_artifact.mojo)
- [kayak/collections/segment.mojo](../../kayak/collections/segment.mojo)
- [kayak/collections/resolved_snapshot.mojo](../../kayak/collections/resolved_snapshot.mojo)
- [docs/late_interaction_2030.md](../late_interaction_2030.md)
- [docs/architecture/native_candidate_generation_next.md](native_candidate_generation_next.md)
- [docs/architecture/gem_engine_integration.md](gem_engine_integration.md)

### Primary or project sources

- Token pooling paper:
  - https://arxiv.org/abs/2409.14683
- ColBERT-Zero paper:
  - https://arxiv.org/abs/2602.16609
- ColBERT-Zero model card:
  - https://huggingface.co/lightonai/ColBERT-Zero-unsupervised
- `pylate-rs` public code/docs:
  - https://github.com/lightonai/pylate-rs

## Verified Repo Facts

These statements are checked against the current repository.

1. Stage-1 generator identity is already explicit and family-aware.
   Evidence:
   - [candidate_generator.mojo](../../kayak/planning/candidate_generator.mojo)
   - built-in families currently include:
     - `exact`
     - `proxy`
     - `centroid`
     - `graph`

2. Stage-1 capability requirements are already separated from generator kind
   strings.
   Evidence:
   - [stage1_capabilities.mojo](../../kayak/planning/stage1_capabilities.mojo)
   - required search artifacts and filter support are now contract-driven

3. Search-visible segment manifests already use a generic search-artifact list
   instead of only one top-level field per engine family.
   Evidence:
   - [segment.mojo](../../kayak/collections/segment.mojo)
   - [search_artifact.mojo](../../kayak/collections/search_artifact.mojo)

4. Snapshot loading is already search-artifact-aware and can load multiple
   artifact families.
   Evidence:
   - [resolved_snapshot.mojo](../../kayak/collections/resolved_snapshot.mojo)

5. Stage-1 execution already dispatches by family module.
   Evidence:
   - [execution.mojo](../../kayak/planning/execution.mojo)

6. Stage 2 is already an explicit part of `SearchPlan`.
   Evidence:
   - [search_plan.mojo](../../kayak/planning/search_plan.mojo)

## Source-Backed Facts About The Referenced Work

### `2409.14683`: token pooling

Directly verified from the paper abstract:

- it introduces a clustering-based token pooling approach
- it targets the storage and memory footprint of multi-vector retrieval
- it claims `50%` space reduction with virtually no retrieval degradation
- it claims larger reductions can still remain under modest degradation on most
  datasets
- it explicitly says the method requires no architectural change and no
  query-time processing, and is meant as a drop-in at indexation time for
  ColBERT-like models

Source:
- https://arxiv.org/abs/2409.14683

Directly verified from public code/docs:

- `pylate-rs` exposes hierarchical pooling at encoding time with a
  `pool_factor`
- the examples use pooling on document embeddings before downstream similarity
  or indexing

Source:
- https://github.com/lightonai/pylate-rs

### `2602.16609`: ColBERT-Zero

Directly verified from the paper abstract:

- the paper studies pre-training for multi-vector models
- it claims that large-scale multi-vector pre-training yields stronger
  multi-vector models
- it explicitly says checkpoints and code are released

Source:
- https://arxiv.org/abs/2602.16609

Directly verified from the model card:

- ColBERT-Zero is presented as a ColBERT / multi-vector model
- it is used through the PyLate ecosystem
- the card points to released checkpoints and training code

Source:
- https://huggingface.co/lightonai/ColBERT-Zero-unsupervised

## Decision

The stable wall for new work in `kayak` should be:

1. encoder and model boundary
2. document-representation transform boundary
3. search-artifact boundary
4. stage-1 engine-family boundary
5. stage-2 refinement boundary
6. stronger-ceiling and evaluation boundary

This is the right wall because it separates three very different kinds of
change that are often mixed together in discussion:

- changing how vectors are produced
- changing how document vectors are stored or transformed
- changing how candidate generation or refinement is executed

## The Six Stable Seams

### 1. Encoder And Model Boundary

What belongs here:

- swapping one ColBERT-like checkpoint for another
- changing prompt templates or query/document prefix discipline
- changing tokenizer or model weights
- changing model-scale or training recipe while preserving compatible
  late-interaction output semantics

What should stay stable below this seam:

- packed-index storage layout
- stage-1 family dispatch
- exact stage-2 reranking logic

What must remain explicit:

- model identity
- vector dimension
- scalar type
- query versus document encoding conventions

### 2. Document-Representation Transform Boundary

What belongs here:

- token pooling
- pruning
- aggregation
- compression-aware document-side transforms
- any change that modifies the stored document representation while preserving
  the late-interaction scoring contract above it

What should stay stable below this seam:

- the fact that a sealed segment stores a search-ready late-interaction
  document representation
- the stage-1 and stage-2 plan contracts

What is still missing in repo terms:

- a first-class way to record document-representation transforms as segment
  metadata rather than only as an implicit property of how one packed index was
  built

Inference:

- this is the main seam `kayak` still needs to make more explicit if it wants
  token-pooling and future pooling/pruning work to plug in cleanly

### 3. Search-Artifact Boundary

What belongs here:

- document proxy sidecars
- centroid postings or centroid heads
- GEM graph artifacts
- future search-native sidecars required by stage 1

What should stay stable below this seam:

- segment manifests name artifacts by family
- loaders resolve those artifacts into typed loaded views

Why this matters:

- adding a new stage-1 sidecar should not force new top-level fields all over
  the segment and snapshot model

### 4. Stage-1 Engine-Family Boundary

What belongs here:

- exact full scan
- proxy-family candidate generation
- centroid-family candidate generation
- graph-family candidate generation
- future families that use different retrieval geometry

What should stay stable below this seam:

- `CandidateGenerator`
- `CandidateSet`
- artifact-family requirements
- family dispatch from `execution.mojo`

What should not happen:

- collapsing different retrieval geometries into one fake "native engine"
  abstraction

### 5. Stage-2 Refinement Boundary

What belongs here:

- exact late interaction reranking
- text-aware reranking
- future hybrid or cross-attention refinement

What should stay stable below this seam:

- stage 1 produces a candidate window
- stage 2 refines that candidate window explicitly

### 6. Stronger-Ceiling And Evaluation Boundary

What belongs here:

- exact full-scan references
- text-aware stronger ceilings
- future expensive verifier or reasoning ceilings

What should stay stable below this seam:

- benchmark reporting
- planner evidence
- explain surfaces

Why this matters:

- stronger ceilings should validate the engine
- they should not silently become the default engine path

## Classification Rule For New Papers

Every proposed integration should answer one question first:

**Which seam is this changing?**

The default expectation should be:

- one paper maps primarily to one seam
- a paper that touches multiple seams is automatically higher-risk

This keeps the repo honest.

If a proposal cannot say whether it is:

- a model swap
- a representation transform
- a search artifact
- a stage-1 engine family
- a stage-2 operator
- or only an evaluation ceiling

then the repo is not ready to implement it yet.

## Applying The Rule To The Two Referenced Papers

### `2602.16609` is primarily an encoder-model integration

Best classification:

- **primary seam:** encoder and model boundary

Why:

- ColBERT-Zero is first and foremost a better multi-vector model and training
  recipe
- it still lives in a ColBERT-like late-interaction setting
- the public ecosystem around it is PyLate/ColBERT-style inference, not a new
  search geometry

Expected blast radius in `kayak`:

- low to medium

What should be needed:

- model manifest discipline
- explicit query/document encoding conventions
- collection compatibility checks
- benchmark reruns

What should not be needed:

- a new stage-1 engine family
- a new stage-2 contract
- a search-manifest refactor

Verdict:

- not trivial in the sense of "drop it in blindly"
- but it **should** be a clean plug-in if the encoder boundary stays explicit

### `2409.14683` is primarily a document-representation transform

Best classification:

- **primary seam:** document-representation transform boundary

Why:

- the paper explicitly presents token pooling as an index-time drop-in
- it does not claim a new query-time retrieval geometry
- it aims to change the stored document representation, not the planner or
  stage topology

Expected blast radius in `kayak`:

- medium

What should be needed:

- transform metadata or provenance at the segment boundary
- vector-budget accounting
- bytes/vector accounting
- exact-stage and final-quality evaluation against an unpooled reference

What should not be needed:

- a new stage-1 family just because pooling is used
- a new stage-2 contract
- a planner rewrite

Verdict:

- not trivial
- but it **should** be much easier than adding a new native candidate engine
  family if the transform seam is kept explicit

## Difficulty Ladder

This is an inference from the current repo state plus the referenced work.

From easier to harder:

1. compatible ColBERT-like model swap
2. document-side representation transform such as pooling
3. new search-native sidecar within an existing family
4. new stage-2 refinement operator
5. new stage-1 engine family with different retrieval geometry
6. a system that changes both representation and retrieval geometry together

This is exactly why the wall matters.

Without it, all six categories collapse into "new paper, refactor the engine".

## What The Repo Should Do Next

The sound next step is not to hardcode token pooling or ColBERT-Zero-specific
paths immediately.

The sound next step is to preserve and sharpen the generic seams:

1. keep model identity explicit at collection and segment boundaries
2. add an explicit document-representation-transform concept before multiple
   pooling/pruning variants arrive
3. keep stage-1 artifact requirements and dispatch family-aware
4. keep stage-2 explicit and candidate-window-scoped
5. require benchmark reports to keep:
   - vectors/document
   - bytes/vector
   - stage-1 recall against exact stage-2 results
   - final quality
   - latency

That sequence is what lets future work plug in without turning every new paper
into a storage and planner rewrite.

## Bottom Line

For the user question "is this trivial to plug in?":

- `2602.16609`:
  - no, not trivial
  - yes, it should be a clean plug-in at the encoder boundary
- `2409.14683`:
  - no, not trivial
  - yes, it should be a clean plug-in at the document-representation boundary
    if that boundary is made explicit

The key result is not "everything is easy".

The key result is narrower and more useful:

- most future late-interaction work should **not** require a core engine
  refactor
- but only if `kayak` keeps the wall explicit and refuses to blur models,
  representation transforms, stage-1 engines, and stage-2 operators together
