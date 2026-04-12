# Late Interaction 2030 Implications

This note records what Kayak should take from Omar Khattab's "Late
Interaction in 2030" workshop talk transcript that was provided in repo
discussion on `2026-04-12`.

This is a strategic interpretation, not a claim that every point from the talk
is already verified in this repository.

The goal is narrower:
- identify which arguments sharpen Kayak's roadmap
- separate repo-supported decisions from still-open research questions
- turn a broad field-level talk into concrete engine work

## What The Talk Argues

The talk makes five strategic claims that matter for Kayak:

1. Retrieval has an unusually large ceiling-floor gap.
   Reason:
   - expensive long-context or cross-attention systems are getting strong
   - scalable first-stage retrieval is still structurally weaker than that
     ceiling on harder tasks

2. Hard retrieval tasks are likely underrepresented.
   Reason:
   - if we assume retrieve-and-rerank up front, we mostly build tasks where
     stage-1 recall is already good enough
   - that can hide cases where reranking cannot rescue weak stage-1 recall

3. Late interaction should be understood as a paradigm, not as one model.
   Reason:
   - the interesting part is local interaction with sublinear search
   - the interesting part is not specifically ColBERT, MaxSim, or one storage
     layout in isolation

4. Infra matters as much as model quality.
   Reason:
   - hosted search products, open search engines, runtime primitives, and
     usable APIs determine adoption
   - algorithm papers alone do not solve updates, filters, serving, or
     observability

5. Benchmarks should compare against stronger ceilings, not only against
   single-vector baselines.
   Reason:
   - the future bar is not just "beat dense retrieval"
   - the future bar is "approach much more expensive reasoning or
     cross-attention quality with a system that scales"

## What Kayak Already Supports

The repository already aligns with part of that framing.

Verified from local docs and code:
- [TODO.md](../TODO.md) already prioritizes retrieval-native storage, explicit
  candidate generation, and serving boundaries ahead of benchmark-specific
  reranker work
- [docs/python_sdk_charter.md](python_sdk_charter.md) already treats late
  interaction as an explicit primitive rather than a generic tensor veneer
- [docs/architecture/service_api.md](architecture/service_api.md) and
  [docs/architecture/segment_storage.md](architecture/segment_storage.md)
  already define engine-facing collection, snapshot, and search-plan contracts

Inference:
- Kayak does not need a strategic rewrite
- it needs a sharper sequence for the next engine milestones

## What Kayak Should Change

The talk does not change Kayak's north star.

It does change how the next steps should be justified:

1. Do not evaluate progress only by beating dense single-vector baselines.
   Instead:
   - keep exact full-scan late interaction as the correctness anchor
   - compare stage-1 candidate generation against exact recall
   - when useful, compare the final recall ceiling against a much more
     expensive long-context or cross-attention path

2. Treat late interaction as local interaction plus sublinear search.
   Therefore:
   - runtime interfaces should stay explicit about layouts, vector counts, and
     stages
   - Kayak should not hide itself behind a generic vector-DB abstraction

3. Finish the engine loop before expanding the benchmark surface much more.
   Therefore:
   - collection ingest, update, snapshot, search, and explain should become one
     coherent hosted-engine path
   - benchmark growth should follow stage contracts rather than replace them

4. Make stage 1 a real product primitive.
   Therefore:
   - exact full scan should remain available
   - one explicit non-exact or pruning generator should exist behind
     `SearchPlan`
   - stage-1 recall must be measurable against exact final results

5. Keep compression and GPU work subordinate to stage contracts.
   Therefore:
   - compression is still important
   - but compression should optimize an explicit serving/storage loop, not
     precede it

## Concrete Next Phases

The next engine sequence for Kayak should be:

### Phase F: Hosted Collection Loop

Goal:
- make the existing service and storage contracts executable end to end

Deliverables:
- create collection
- ingest and upsert documents
- delete documents
- snapshot and restore
- search and explain against a hosted collection

Exit criteria:
- an external user can run one exact late-interaction collection without
  benchmark-specific fixtures

### Phase G: Native Stage-1 Candidate Generation

Goal:
- move from exact-only serving to an explicit stage-1 plus stage-2 engine

Deliverables:
- exact full-scan candidate generator
- one pruning or approximate candidate generator
- candidate-set tracing and stage-level profiling
- recall reporting against exact final results

Exit criteria:
- every result set can say which stage produced which candidates and what stage
  1 recall it achieved against exact

### Phase H: Hard-Recall Evaluation

Goal:
- benchmark the engine on tasks where weak stage-1 recall is the real problem

Deliverables:
- one or two harder public or synthetic tasks selected for low-recall pressure
- ceiling comparisons against a much more expensive path when justified
- benchmark outputs that distinguish:
  - stage-1 recall
  - final retrieval quality
  - latency and storage tradeoffs

Exit criteria:
- Kayak can show why its engine design matters on tasks where reranking alone is
  not enough

## What This Does Not Mean

This note does not imply:
- that ColBERT exactly as-is is the final answer
- that Kayak should pivot to training models first
- that Kayak should add many more benchmark slices immediately
- that the next milestone should be a generic text reranker
- that the next milestone should be GPU or distributed sharding

Those are still downstream of storage, stage, and serving clarity.

## Decision Rule

If a proposed roadmap item helps Kayak become:
- a clearer hosted late-interaction engine
- a better stage-aware retrieval system
- a stronger benchmarked substrate for hard-recall tasks

then it likely belongs near the top of the roadmap.

If it mainly:
- decorates one benchmark
- hides late-interaction semantics
- avoids explicit stage boundaries
- or duplicates what a generic vector store already does

then it likely belongs lower.
