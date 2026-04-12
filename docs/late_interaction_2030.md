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

Verified since the first version of this note:
- the hosted collection loop is now executable
- non-exact stage-1 generators now sit behind explicit `SearchPlan` contracts
- hard-recall benchmark output now reports candidate recall against an exact
  full-scan reference

Those are substrate wins, not proof of the stronger efficiency thesis.

Verified after the first version of this note:
- there is now one synthetic single-core scaling benchmark over increasing
  corpus sizes
- that benchmark already shows a measurable latency-recall tradeoff between
  exact full scan, `document_proxy`, and capped native candidate engines

Evidence:
- [docs/traces/2026-04-12_single_core_scale.md](traces/2026-04-12_single_core_scale.md)

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

## What The Repo Still Has Not Proved

The strongest claims implied by the talk are still open.

Not yet locally verified:
- under-`200ms` single-core search at multi-billion-token scale
- token storage close to `6 bytes/vector`
- vectors/document pruning laws anywhere near `sqrt(m)` in the regimes we care
  about
- asymptotic native-engine scaling on harder or larger benchmark families well
  beyond the current synthetic sweep
- a benchmark family clearly harder than today's small public slices
- a stronger expensive ceiling that is implemented and compared locally

That gap matters because it is now easy to confuse "the repo has the right
substrate" with "the repo has already proved the efficiency thesis."

## Parallel Roadmap

The right next step is not another rewrite of the engine sequence.

It is to run two tracks in parallel:
- keep the engine/product substrate moving
- start an explicit efficiency-and-scaling track that tries to prove or debunk
  the stronger late-interaction claims

That roadmap is recorded in:
- [docs/late_interaction_efficiency_roadmap.md](late_interaction_efficiency_roadmap.md)

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
- a more measurable testbed for efficiency and scaling claims

then it likely belongs near the top of the roadmap.

If it mainly:
- decorates one benchmark
- hides late-interaction semantics
- avoids explicit stage boundaries
- or duplicates what a generic vector store already does

then it likely belongs lower.
