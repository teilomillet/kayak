# Epistemic Standard For The Course

This note defines the minimum evidence bar for internal course material.

The course is allowed to be opinionated.
It is not allowed to be sloppy.

## The Failure Mode To Avoid

The main risk is:

- teaching strong retrieval conclusions from a nice demo that does not justify
  them

Typical bad examples:

- claiming a general latency win from one noisy notebook run
- claiming broad retrieval superiority from one hand-built toy example
- using a benchmark trace without stating what it actually measured
- teaching chunking advice as universal when the repo only verified it on one
  workload family

There is also a second failure mode:

- turning the course into an explicit methods lecture when the learner really
  wants to poke at retrieval, notice what breaks, and understand why

The standard is therefore partly backstage.
It exists to keep our claims honest.
It does not require the learner to constantly think in status labels.

## Status Labels

Every non-trivial claim in a lesson should carry one of these statuses.

### Verified Locally

Use this only when:

- the current repository contains a runnable reproduction
- we reran that reproduction in this work or the lesson points to an existing
  test with the same claim

Examples:

- a deterministic unit test
- a notebook whose core claim is covered by a small smoke test
- a benchmark command with recorded artifact output in this repo

### Supported On A Defined Surface

Use this when:

- the claim holds on named workloads or traces
- the evidence is real but intentionally scoped

Examples:

- public-slice hard-recall traces
- one task-comparison bundle
- one measured storage handoff workflow

This status must name the surface explicitly.

### Narrowly Falsified

Use this when the repo has direct counter-evidence against a stronger version of
the claim.

This is useful course material.
It teaches learners where intuition breaks.

### Instrumented But Unresolved

Use this when:

- the repo can measure the question
- the evidence is not yet broad or stable enough to teach as settled

This status belongs in advanced lessons and instructor notes, not in marketing.

## Minimum Lesson Shape

Every lesson should include:

- `Problem`
- `What The Learner Will Be Able To Do`
- `Primitives Introduced`
- `Claims And Evidence`
- `Rerun Commands`
- `Where This Does Not Yet Generalize`

Reason:
- this makes the course legible to researchers, product people, and engineers
- it also prevents the lesson from hiding its weak spots

For internal production, this shape should stay explicit.
For learner-facing delivery, parts of it can be implicit as long as the same
truth constraints are preserved.

## Notebook Rules

Notebooks are allowed to use toy data.
Toy data must not be sold as benchmark evidence.

For notebook claims:

- clearly label whether the notebook is a toy reproduction, a smoke path, or a
  benchmark reproduction
- keep the core claim mechanically covered by a small automated test when
  practical
- separate deterministic rank-order checks from exploratory timing cells

For notebook delivery:

- let the learner play first when possible
- surface the epistemic label when a strong claim, metric, or recommendation is
  about to be made
- do not force the learner through evidence bookkeeping before they have felt
  the practical problem

Reason:
- notebook timing is usually too noisy to carry the main argument
- deterministic rank-order checks are much better teaching anchors

## Performance Claim Rules

No lesson should claim a speedup unless it states:

- the workload shape
- the backend
- the index layout
- whether the run was a microbenchmark, smoke path, or end-to-end path

And no lesson should imply that:

- a local speedup automatically generalizes to all corpora
- one batch microbenchmark proves service-level latency behavior

## Course-Worthy Claims

These are examples of claims that fit the standard well:

- late interaction can recover an evidence-bearing document that a mean-pooled
  dense baseline misses on a deterministic toy example
- stage-1 recall pressure matters on the repo's hard-recall benchmark surfaces
- reusing one loaded index is the public repeated-query fast path in the Python
  SDK
- Kayak is currently far from the public frontier on at least one full benchmark

These are examples of claims that do not yet fit:

- Kayak solves RAG in general
- late interaction is always better than dense retrieval
- the current engine is ready for every production workload

## Delivery Implication

The course should therefore feel like:

- "try this"
- "notice what changed"
- "here is the variable you just changed"
- "here is the narrow thing we now know"

It should not feel like:

- "before we touch anything, here are six epistemic labels"
- "please think like a benchmark author before you think like a user"
