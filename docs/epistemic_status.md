# Epistemic Status Of 2030 Claims

Date: `2026-04-13`

This note records the current epistemic status of the strongest
late-interaction claims that matter for `kayak`.

It exists to prevent a specific failure mode:

- confusing substrate progress with proof of the broader efficiency thesis

This document is intentionally narrower than a manifesto.

The goal is to make each major claim legible as:

- what the claim actually says
- what would count as proof
- what would count as falsification
- what the repository currently measures
- what verdict is justified today

## Sources Checked

- [TODO.md](../TODO.md)
- [docs/late_interaction_2030.md](late_interaction_2030.md)
- [docs/late_interaction_efficiency_roadmap.md](late_interaction_efficiency_roadmap.md)
- [docs/hard_recall_evaluation.md](hard_recall_evaluation.md)
- [docs/traces/2026-04-12_single_core_scale.md](traces/2026-04-12_single_core_scale.md)
- [docs/traces/2026-04-12_single_core_faithfulness_frontier.md](traces/2026-04-12_single_core_faithfulness_frontier.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md](traces/2026-04-12_browsecomp_plus_gold_faithfulness_frontier.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_storage_encoding.md](traces/2026-04-12_browsecomp_plus_gold_storage_encoding.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md](traces/2026-04-12_browsecomp_plus_gold_vector_pruning.md)
- [docs/traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md](traces/2026-04-12_browsecomp_plus_gold_ceiling_comparison.md)
- [docs/traces/2026-04-12_synthetic_hard_recall_stage_aware.md](traces/2026-04-12_synthetic_hard_recall_stage_aware.md)
- [docs/traces/2026-04-13_centroid_primitives.md](traces/2026-04-13_centroid_primitives.md)
- [docs/traces/2026-04-13_hosted_engine_p0.md](traces/2026-04-13_hosted_engine_p0.md)
- [docs/traces/2026-04-13_hosted_engine_p1_ops.md](traces/2026-04-13_hosted_engine_p1_ops.md)
- [docs/traces/2026-04-13_exact_doc_id_filters.md](traces/2026-04-13_exact_doc_id_filters.md)

## Status Key

- `Verified implementation claim`
  - the repository proves a concrete property of a specific implementation
- `Supported on a defined surface`
  - current evidence points in one direction on named workloads
- `Narrowly falsified`
  - a strong form of the claim fails on at least one defined surface
- `Instrumented but unresolved`
  - the repo can measure the claim, but current evidence is not broad enough
- `Unverified broad claim`
  - the repo does not yet justify the field-level statement
- `Deferred`
  - important, but intentionally not the current decision driver

## Ground Rules

Two rules keep this document honest:

1. We distinguish implementation claims from field claims.
   Example:
   - "this storage format reaches `256.08 bytes/vector` on this slice" is an
     implementation claim
   - "late interaction works at `~6 bytes/vector`" is a broad field claim

2. Empirical evidence is always conditional on:
   - the model
   - the storage format
   - the benchmark family
   - the pruning or candidate-generation policy

That means the repo can prove many local statements while still not proving the
broader 2030 thesis.

## Claim 1: Stage-1 Recall Pressure Matters

Claim:

- hard retrieval tasks exist where weak stage-1 recall is the real bottleneck,
  and reranking cannot rescue a poor first-stage shortlist

What would count as proof:

- stage-aware benchmarks where exact-reference candidate recall and final
  retrieval quality are reported separately
- public or synthetic tasks where non-exact stage-1 plans materially lose
  recall before exact reranking happens

What would count as falsification:

- broad evidence that realistic candidate generators preserve exact-reference
  recall so reliably that stage-1 recall rarely constrains final quality

What the repo currently measures:

- explicit stage-1 recall against exact full-scan references
- public hard-recall slices for `BrowseComp-Plus` evidence and gold
- scalable synthetic hard-recall families

Current verdict:

- `Supported on a defined surface`

Why:

- this claim is no longer speculative inside `kayak`
- the public and synthetic stage-aware artifacts show meaningful recall loss
  for several non-exact plans before exact reranking recovers

What is still missing:

- a broader public family beyond today's small public slices

## Claim 2: A `sqrt(m)`-Style Vector Budget Is Plausible

Claim:

- the number of document vectors needed for useful late interaction may grow
  much more slowly than linearly with document length, potentially near a
  `sqrt(m)`-style law

What would count as proof:

- multiple pruning or aggregation policies measured over multiple benchmark
  families
- near-`sqrt(m)` vector budgets that preserve acceptable stage-1 recall and
  final retrieval quality across those families

What would count as falsification:

- repeated evidence across multiple strong pruning policies and workloads that
  near-`sqrt(m)` budgets collapse exact-reference recall or judged quality

What the repo currently measures:

- one direct vectors/document benchmark on BrowseComp gold using naive prefix
  pruning

Current verdict:

- `Narrowly falsified`

Why:

- the strong naive version of the claim already fails on a meaningful local
  slice
- on BrowseComp gold:
  - full average vectors/document is about `175.07`
  - `sqrt(175.07)` is about `13.23`
  - budget `16` preserved only `62.5%` of full exact top-`10`
  - judged quality also dropped sharply

What this does not mean:

- it does **not** falsify all possible vector-budget laws
- it falsifies the naive prefix-pruning story on the measured slice

What remains open:

- whether better pruning, clustering, pooling, or architecture-aware token
  selection can rescue a much smaller vector budget

## Claim 3: Late Interaction Can Reach `~6 Bytes/Vector`

Claim:

- late-interaction storage can be pushed toward a very small bytes/vector
  regime without unacceptable retrieval loss

What would count as proof:

- one or more explicit storage formats reaching that regime in `kayak`
- measured bytes/vector, latency, and retrieval-quality tradeoffs on defined
  hard-recall slices

What would count as falsification:

- repeated evidence that storage near that regime causes unacceptable quality
  collapse or intolerable decode/latency overhead across multiple reasonable
  formats

What the repo currently measures:

- one compressed packed-index path:
  - `binary_f16_le`

Current verdict:

- `Instrumented but unresolved`

Why:

- the repo has one concrete compression result, not the full claim
- on BrowseComp gold, `binary_f16_le` reduced persisted bytes/vector from about
  `512.08` to `256.08` with no measured retrieval-quality change
- that is evidence for one better storage point, not evidence for a `~6
  bytes/vector` regime

What remains open:

- lower-bit or codebook-based token storage
- in-memory compressed scoring rather than decode-on-load formats
- quality and latency behavior at far smaller byte budgets

## Claim 4: Native Candidate Engines Have Stronger Practical Asymptotics

Claim:

- native stage-1 late-interaction engines can scale substantially better than
  naive full-corpus scoring in practice, not just on paper

What would count as proof:

- explicit scaling curves over increasing corpus sizes
- exact-reference recall and final quality reported with the same runs
- wins that remain meaningful as corpus size grows, not only on one tiny slice

What would count as falsification:

- repeated evidence that plausible native engines fail to beat exact full scan
  once comparable recall is required on representative workloads

What the repo currently measures:

- one synthetic single-core scaling benchmark
- one synthetic faithfulness frontier
- one BrowseComp gold frontier mirror

Current verdict:

- `Instrumented but unresolved`

Why:

- the repo can now measure this claim honestly
- but current evidence is mixed:
  - on synthetic slices, some native paths recover full recall cheaply
  - on the current small BrowseComp gold slice, `document_proxy` is the
    cheapest measured full-recall point, while current centroid-family plans do
    not beat exact latency once they approach full recall

What this means:

- the broad asymptotic thesis is not proved
- some candidate-engine families are promising
- others are not yet justified on the harder public slice

## Claim 5: A Broad Stronger Local Ceiling Exists And Matters

Claim:

- the right comparison bar is not only dense retrieval or exact MaxSim, but a
  much stronger and more expensive local ceiling such as richer reranking,
  cross-attention, or long-context reasoning

What would count as proof:

- one implemented local ceiling path that is measurably stronger than the
  current exact late-interaction baseline on more than one narrow slice
- explicit cost reporting for that stronger path

What would count as falsification:

- evidence that no stronger local path is meaningfully above exact late
  interaction on the tasks that matter for `kayak`

What the repo currently measures:

- one local stronger comparison on BrowseComp gold:
  - exact full scan plus `clause_text` reranking

Current verdict:

- `Supported on a defined surface`

Why:

- the stronger local ceiling exists in at least one narrow local form
- on BrowseComp gold, it can outperform exact MaxSim
- it is also much more expensive

Why this is still not enough:

- the ceiling is not yet broad
- it is not yet the default benchmark reference for stage-aware evaluation

## Claim 6: The Repo Has Already Proved The Strong 2030 Efficiency Thesis

Claim:

- `kayak` already proves the main stronger 2030 claims about scaling,
  compression, vector budgets, and stronger ceilings

What would count as proof:

- strong positive verdicts across Claims 2 through 5 above, on sufficiently
  broad benchmark and implementation surfaces

What would count as falsification:

- exactly the current situation: mixed evidence, narrow wins, and open claims

Current verdict:

- `Unverified broad claim`

Why:

- the repo has the right substrate
- the repo has several important local measurements
- the repo does **not** yet justify the full stronger efficiency thesis

This is the most important status line in the document.

## Claim 7: Hosted-Engine Continuity Is A High-Confidence Next Step

Claim:

- continuing to strengthen the hosted late-interaction engine is valuable even
  while the broader 2030 efficiency claims remain unresolved

What would count as proof:

- the work advances storage, serving, updates, filters, explainability,
  profiling, or tenant isolation in ways that remain useful under multiple
  future stage-1 outcomes

What would count as falsification:

- evidence that hosted-engine work is only useful if one specific candidate
  engine family wins

What the repo currently supports:

- hosted collection create, mutate, snapshot, export/import, search, and
  explain
- explicit `SearchPlan` stages
- persistent search-native sidecars per sealed segment

Current verdict:

- `Verified implementation claim`

Why:

- hosted-engine continuity is robust to uncertainty about:
  - centroid versus GEM-style native engines
  - future compression paths
  - future larger hard-recall families
  - future stronger ceiling choices

Inference:

- this is the highest-confidence focus while broader research claims remain
  open

## Decision Rule

When evaluating the next milestone, use this order:

1. Prefer work that remains valuable even if Claims 2 through 5 are revised.
2. Prefer claims that can be falsified locally over slogans that cannot.
3. Do not upgrade a broad claim because one implementation variant improved.
4. Keep the verdict vocabulary strict:
   - verified here
   - supported here
   - narrowly falsified here
   - unresolved

That is the standard this repo should use before saying a 2030 claim is
"proved."
