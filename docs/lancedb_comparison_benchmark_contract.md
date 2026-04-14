# LanceDB Comparison Benchmark Contract

Status: working comparison contract  
Date: `2026-04-14`

This note defines the narrow benchmark contract that Kayak should use for any
future claim of being "better than LanceDB" on a retrieval surface.

It is intentionally strict.

Reason:
- LanceDB is already a real search engine, not just a storage layer
- the repo already has several useful internal benchmark surfaces
- without a fixed external-comparison contract it would be too easy to
  cherry-pick one lane, one metric, or one operating point

This note therefore does three things:
- defines the benchmark lanes that matter
- defines what counts as a win
- defines what each failure mode means for optimization

## Sources Checked

Local repo sources:
- [docs/benchmark_ladder.md](benchmark_ladder.md)
- [docs/benchmark_rationale.md](benchmark_rationale.md)
- [docs/hard_recall_evaluation.md](hard_recall_evaluation.md)
- [docs/epistemic_status.md](epistemic_status.md)
- [docs/late_interaction_2030.md](late_interaction_2030.md)
- [docs/architecture/data_flow_io.md](architecture/data_flow_io.md)

Official LanceDB sources checked on `2026-04-14`:
- Quickstart: https://docs.lancedb.com/quickstart
- Search overview: https://docs.lancedb.com/search
- Multivector search: https://docs.lancedb.com/search/multivector-search
- Metadata filtering: https://docs.lancedb.com/search/filtering
- Reranking: https://docs.lancedb.com/reranking

Verified from those official docs:
- LanceDB already exposes vector, multivector, full-text, hybrid, filtering,
  and reranking surfaces
- LanceDB can run embedded or against object-store-backed data
- LanceDB is therefore a legitimate retrieval baseline, not just a file format

## Verified Starting Point

Verified from the local repository:
- Kayak already has an explicit exact full-scan late-interaction reference path
- Kayak already has stage-aware public hard-recall output on
  `BrowseComp-Plus` gold and evidence slices
- Kayak already has scalable synthetic hard-recall families with explicit
  vector-count control and exact-reference candidate recall reporting
- the repo already treats LanceDB-style systems as potential adapters or
  substrates rather than as Kayak's canonical semantic contract

Important boundary:
- this note does **not** assume Kayak must use LanceDB search
- it only assumes LanceDB is a serious external comparison target

## The Claim Boundary

Kayak should **not** make the broad claim:

- "Kayak is better than LanceDB"

unless it first satisfies the narrower claim:

- "Kayak is better than LanceDB on the benchmark contract defined in this note"

Reason:
- that keeps the claim conditional on explicit workloads, metrics, and
  operating envelopes
- it prevents field-level overclaiming from one good result

## Comparison Modes

Every external comparison should run in two modes.

### Mode 1: Product-Fair

Each system uses the strongest retrieval configuration that is natural for that
system on the selected lane.

For LanceDB that may include:
- vector search
- multivector search
- hybrid search
- reranking where the lane definition permits it

For Kayak that may include:
- exact late interaction
- one explicit non-exact native stage-1 engine
- one stronger local ceiling where the lane definition permits it

Reason:
- this is the fairest way to answer the buyer question
- users compare products in their strongest intended modes, not in artificial
  handicaps

### Mode 2: Systems-Fair

Both systems operate on the same encoded corpus and query representations with
the same judged query set and the same hardware envelope.

Reason:
- this isolates where the win comes from
- it prevents one side from winning only because it used a different encoder or
  an easier storage boundary

Zero-vector rule:
- if one system requires zero vectors to be filtered for the selected metric,
  then the task must be filtered once up front and that same filtered task must
  be used for every branch in the comparison

Reason:
- otherwise one branch is paying for vectors that the other branch has removed
- the threshold study showed this is large enough to distort the apparent
  crossover point

## Benchmark Lanes

Kayak should use three lanes, not one.

### Lane A: Public Hard Retrieval

Current anchor:
- `BrowseComp-Plus` gold slice

Question this lane answers:
- can Kayak beat LanceDB on a public hard retrieval slice where stage-1 recall
  and exact evidence matter more than shallow nearest-neighbor similarity?

Required reported metrics:
- primary judged metric on the slice
- exact-reference candidate recall at final `k`
- mean non-debug search latency
- bytes per document
- bytes per vector
- mean document vectors per document

Reason:
- this is the best current repo-supported hard public lane
- it is already stage-aware inside Kayak

### Lane B: Externally Legible Public Lane

Current required target:
- one multilingual public lane

Preferred first candidate:
- `MIRACL`

Reason:
- `MIRACL` is already named in the repo's benchmark rationale
- multilingual search is externally legible and easy to explain
- it gives Kayak a public lane that is not only "hard evidence retrieval"

Current epistemic status:
- required next lane
- not yet the current repo's external-comparison anchor

Important boundary:
- a biology or biomedical lane may become the stronger follow-on story, but it
  should not replace this lane until the dataset and loader path are audited
  explicitly

### Lane C: Scalable Diagnosis Lane

Current anchors:
- `synthetic_hard_recall`
- `long_document_hard_recall`

Question this lane answers:
- if Kayak wins or loses on public lanes, what exactly is failing in the
  engine: stage-1 recall, long-document robustness, latency scaling, or byte
  cost?

Required reported metrics:
- exact-reference candidate recall at final `k`
- mean non-debug search latency
- stage-1 latency
- bytes per vector
- vectors per document
- frontier behavior as `candidate_k` moves

Reason:
- this lane is not for public marketing first
- it is the shortest path to understanding what to optimize next

## Required Baselines

For every lane, record all three of these:

1. `Kayak exact`
   - exact full-scan late interaction
   - this is Kayak's correctness anchor

2. `Kayak best approximate`
   - the current strongest non-exact Kayak plan on that lane
   - this is Kayak's optimization target

3. `LanceDB best allowed`
   - the best LanceDB configuration found under the lane's allowed comparison
     budget

Reason:
- a Kayak approximate result without Kayak exact is not interpretable
- a LanceDB result without a fixed tuning budget is too easy to manipulate

## Tuning Budget Rule

For each system and lane, the comparison must declare a fixed tuning budget
before the final numbers are selected.

Minimum declaration:
- which modes were tried
- which knobs were tuned
- how many configurations were evaluated
- which final configuration was selected and why

For stochastic or rebuild-sensitive external indexes, the declaration must also
say:
- whether the reported result is from one build, the mean across repeated
  rebuilds, the median across repeated rebuilds, or best-of-`N`
- the exact rebuild budget `N`
- the min and max observed values for the primary metric and latency when
  variance is material

Reason:
- this is the smallest guardrail against cherry-picking
- it makes re-runs and backtesting possible

## Win Rules

Kayak can make three different kinds of statements.

### 1. Internal Optimization Win

Allowed when:
- Kayak best approximate improves over the previous Kayak best approximate on
  at least one lane
- and Kayak exact remains the same correctness anchor

This is an internal engineering statement, not an external product claim.

### 2. Lane Win Against LanceDB

Kayak wins a lane only if one of the following is true:

#### Quality-Dominant Win

- Kayak's primary quality metric is at least `5%` relatively better than
  LanceDB's
- and Kayak latency is not more than `2x` LanceDB latency
- and Kayak bytes/document is not more than `2x` LanceDB bytes/document

Reason:
- a small quality wiggle is not enough for a public claim
- a `10x` slower result is a research result, not a product win

#### Efficiency-Dominant Win

- Kayak's primary quality metric is within `1%` relative of LanceDB's
- and Kayak latency is at least `20%` better
- and Kayak bytes/document is no worse than `25%` above LanceDB's

Reason:
- near-tied quality can still justify a product win if the system is
  materially cheaper or faster

### 3. Narrow Public Claim

Kayak may publicly say:

- "Kayak is better than LanceDB on our defined benchmark contract"

only if:
- Kayak wins Lane A
- Kayak wins Lane B
- and Lane C explains the win without a hidden red flag

The required Lane C sanity condition is:
- Kayak's chosen approximate plan must keep exact-reference candidate recall at
  or above `95%` at the selected operating point

Reason:
- this prevents a win that depends on a brittle or opaque candidate pipeline
- it keeps the story centered on strong retrieval, not accidental reranking

## Red Flags That Block A Claim

Even if one headline metric looks good, Kayak should **not** claim a win if:
- Kayak loses the primary quality metric badly on either public lane
- Kayak only wins through a much stronger reranking path that LanceDB was not
  allowed to match in the same comparison mode
- Kayak approximate candidate recall at the chosen point is below `95%`
- Kayak storage or latency exceeds the `2x` envelope without a very large
  quality margin

## Backtesting Contract

Every comparison run that matters should record:
- git SHA for Kayak
- LanceDB version
- dataset slice identifier
- encoder identity
- hardware summary
- tuning budget
- final chosen configs
- all reported metrics

The backtesting rule is:
- do not compare a new Kayak result only to memory
- compare it to:
  - the previous best Kayak result on the same lane
  - the frozen LanceDB baseline on the same lane

Reason:
- this makes progress cumulative
- it prevents "we got a nice run once" from becoming product truth

## What To Optimize When A Lane Fails

This is the most important operational part of the contract.

### If public quality is down and candidate recall is down

Optimize:
- stage-1 engine quality
- candidate budget policy
- search artifacts
- document representation transforms

Interpretation:
- the failure is in retrieval before reranking or exact refinement can help

### If public quality is down but candidate recall is high

Optimize:
- exact scoring path
- representation quality
- query/document encoding assumptions
- optional stronger verifier path

Interpretation:
- stage 1 is probably not the main bottleneck

### If quality is good but latency is bad

Optimize:
- storage layout
- kernel efficiency
- scan avoidance
- compression/decode behavior
- stage-1 pruning efficiency

Interpretation:
- semantics are working; systems work is lagging

### If quality and latency are good but bytes are bad

Optimize:
- storage encoding
- vectors/document
- representation compression

Interpretation:
- the engine works, but the deployment story is too expensive

### If synthetic lanes improve but public lanes do not

Interpretation:
- the engine is optimizing to the wrong failure mode

Action:
- revisit the public lane before claiming progress

### If public lanes improve but synthetic lanes do not

Interpretation:
- the system may have found a narrow public win without scalable stage-1
  strength

Action:
- keep the public result
- do not oversell it as a general stage-1 breakthrough

## Immediate Focus Target

Before any broad public comparison claim, Kayak should do the following:

1. Freeze Lane A on `BrowseComp-Plus` gold with a LanceDB baseline.
2. Add Lane B as one multilingual public lane, preferably `MIRACL`.
3. Keep Lane C as the diagnosis lane for every optimization cycle.
4. Version and store the scorecard so every new run is compared to:
   - previous Kayak best
   - current LanceDB best

That is the minimum contract that can support disciplined optimization and
defensible communication.

## What This Note Does Not Claim

This note does not claim:
- that Kayak already beats LanceDB
- that `BrowseComp-Plus` gold alone is enough for that claim
- that a biology lane is rejected
- that LanceDB must be used as Kayak's storage substrate

It only defines the sound comparison contract:
- explicit lanes
- explicit win rules
- explicit backtesting
- explicit diagnosis of what to optimize next
