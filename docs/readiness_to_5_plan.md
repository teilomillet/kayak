# Readiness To 5 Plan

Status: working completion plan  
Date: `2026-04-13`

This note defines the plan to move `kayak` from its current state to a
defensible `5/5` on every adoption-critical axis.

This is intentionally epistemic:
- it separates what the repository already proves from what is still missing
- it separates repo-controlled work from product decisions and external
  validation
- it defines completion by exit criteria, not by intuition

Related notes:
- [docs/product_positioning.md](product_positioning.md)
- [docs/architecture/service_api.md](architecture/service_api.md)
- [docs/python_sdk_charter.md](python_sdk_charter.md)
- [docs/epistemic_status.md](epistemic_status.md)
- [docs/benchmark_ladder.md](benchmark_ladder.md)
- [docs/hosted_engine_grand_plan.md](hosted_engine_grand_plan.md)
- [TODO.md](../TODO.md)

## Sources Checked

Checked before writing this plan:

- [TODO.md](../TODO.md)
- [docs/product_positioning.md](product_positioning.md)
- [docs/architecture/service_api.md](architecture/service_api.md)
- [docs/python_sdk_charter.md](python_sdk_charter.md)
- [docs/epistemic_status.md](epistemic_status.md)
- [docs/benchmark_ladder.md](benchmark_ladder.md)
- [docs/hosted_engine_grand_plan.md](hosted_engine_grand_plan.md)

Reason:
- these documents already define the current product scope, technical
  contracts, epistemic limits, and open roadmap items

## What "5/5" Means

For this plan, `5/5` does **not** mean:
- "perfect forever"
- "fastest on every imaginable workload"
- "feature-complete compared to every vector database"

It means:
- a company can understand what Kayak is
- a company can deploy it with clear expectations
- the retrieval semantics are explicit and testable
- the operational path is documented and validated
- the performance claims are backed by decision-grade evidence

## Current Score Snapshot

This is the current working assessment.

- Engine semantics: `4.5/5`
- Hosted engine contracts: `4/5`
- Python SDK direction: `3.5/5`
- Benchmark and evidence discipline: `4/5`
- Performance proof against the market: `2/5`
- Deployment trust and peace of mind: `2/5`
- Data boundary and privacy clarity: `2.5/5`
- Competitive category clarity: `3.5/5`

Interpretation:
- the semantic substrate is already strong
- the deployable product surface is still incomplete

## Completion Axes

The plan is complete only when each axis reaches its own exit criteria.

### Axis 1: Engine Semantics

Target:
- explicit and stable late-interaction semantics across SDK, planner, service,
  and benchmark surfaces

Current status:
- close to complete

What still blocks `5/5`:
- remaining encoder-boundary ambiguity
- need for one explicit mixed-model rejection path

### Axis 2: Hosted Engine Contracts

Target:
- typed contracts plus a real deployable service path

Current status:
- strong typed contracts, but no finished service transport or deployment UX

What still blocks `5/5`:
- no networked service path yet
- no self-hosted operator-facing flow yet

### Axis 3: Python SDK

Target:
- `pip install kayak` yields a predictable local developer experience with
  explicit backend behavior

Current status:
- direction is good, friction still exists

What still blocks `5/5`:
- runtime and packaging clarity
- backend discovery and fallback polish

### Axis 4: Performance Proof

Target:
- one credible benchmark pack that external users can rerun and trust

Current status:
- measurement surfaces exist

What still blocks `5/5`:
- insufficient decision-grade comparative evidence

### Axis 5: Deployment Trust

Target:
- clear operator story for deploy, observe, recover, and upgrade

Current status:
- contracts exist, operator path is incomplete

What still blocks `5/5`:
- transport, auth, deployment guides, integration tests, runbooks

### Axis 6: Data Boundary And Privacy Clarity

Target:
- explicit default trust model with enforcement where possible

Current status:
- category note exists, hard product boundary is not fully written or enforced

What still blocks `5/5`:
- no default trust-model note
- no explicit operator-facing data handling contract

### Axis 7: Competitive Category Clarity

Target:
- outsiders can understand why Kayak exists relative to Chroma, Qdrant,
  Firnflow, and ColBERT

Current status:
- improving

What still blocks `5/5`:
- story has not yet been pushed into all public entrypoints

## Ordered Plan

The steps below are ordered by dependency and leverage.

## Step 1: Lock The Trust And Data Boundary

Why first:
- "peace of mind" depends on what Kayak stores, what it does not store, and
  what the default deployment boundary means
- without this, later deployment or security claims remain fuzzy

Required work:
- [ ] Write one explicit trust-model note
- [ ] Define the default storage boundary:
  - raw documents
  - vectors
  - metadata
  - optional text sidecars
- [ ] Define customer-side vs service-side encoding modes
- [ ] State precisely what privacy claims Kayak can and cannot make
- [ ] Add service-facing language for vector sensitivity

Exit criteria:
- one note exists and is linked from the public entrypoints
- the service and product docs use the same boundary language
- the repo no longer implies "Kayak never sees user data" when vectors or
  metadata are stored

## Step 2: Finish Priority 9 Cleanly

Why next:
- one encoder space per collection is a correctness and product-trust issue,
  not just a research opinion

Required work:
- [ ] Add one design note for multi-encoder interoperability and non-goals
- [ ] Keep collection and snapshot contracts explicit about one encoder space
- [ ] Add one negative test rejecting unsound mixed-model assumptions

Exit criteria:
- mixed-model behavior is documented as unsupported by default
- a validation path rejects unsafe mixing when surfaced
- the SDK and service docs do not imply model-agnostic late interaction

Source:
- open TODOs already exist in [TODO.md](../TODO.md)

## Step 3: Turn The Typed Service Contract Into A Real Service

Why:
- typed contracts are necessary but not sufficient for adoption
- companies adopt a deployable system, not only a good set of structs

Required work:
- [ ] Implement one real service transport around `kayak/service/`
- [ ] Add one networked end-to-end integration test:
  create collection, ingest, snapshot, search, explain
- [ ] Expose health and metrics through the real service path
- [ ] Add one local development startup path

Exit criteria:
- an external user can run one service process and hit it over the network
- the documented flow does not require reading repo internals
- integration tests cover the main happy path

## Step 4: Make Self-Hosted Deployment Boring

Why:
- ease of deployment is where Chroma and Qdrant feel safe
- Kayak needs one operator story that is simple and explicit

Required work:
- [ ] Write one self-hosted deployment guide
- [ ] Define required services and runtime dependencies
- [ ] Add one Docker or equivalent deploy path
- [ ] Document persistence, snapshot roots, and recovery assumptions
- [ ] Add a startup, readiness, and shutdown checklist

Exit criteria:
- a new user can self-host Kayak without reading source files
- the service can be started, probed, and stopped predictably
- the data and snapshot paths are explicit

## Step 5: Add The Operator Safety Layer

Why:
- companies need more than "search works"
- they need to know how it behaves under failure, retention, and upgrades

Required work:
- [ ] Document snapshot and restore workflow
- [ ] Document reclaim and compaction workflow
- [ ] Add one failure-mode note:
  - process crash during seal
  - process crash during publish
  - process crash during reclaim
- [ ] Add one upgrade and rollback note
- [ ] Add one retention-policy operator guide

Exit criteria:
- there is a documented recovery story for the main lifecycle operations
- snapshot, reclaim, and restore are no longer only code-level capabilities

## Step 6: Add Auth And Tenant-Isolation At The Service Edge

Why:
- tenant isolation in manifests is not enough for external trust
- the service edge needs a real story for identity and authorization

Required work:
- [ ] Define one minimal auth model
- [ ] Define how requests map to tenant and namespace scope
- [ ] Ensure service requests cannot bypass logical scope contracts
- [ ] Add integration tests for tenant isolation at the network boundary

Exit criteria:
- the service edge has a documented auth model
- tenant isolation is enforced above the engine internals

## Step 7: Finish The Python SDK Experience

Why:
- `pip install kayak` is a core part of the public story
- the SDK must feel predictable, not toolchain-fragile

Required work:
- [ ] Publish one clear install matrix:
  - NumPy reference
  - Mojo exact CPU
  - optional PyTorch input path
- [ ] Make backend selection and backend info easy to inspect
- [ ] Document graceful fallback when Mojo is unavailable
- [ ] Add one fresh-environment validation matrix for:
  - `pip`
  - `uv`
  - `pixi`
- [ ] Add one simple remote-engine client story if Kayak Engine becomes
  networked

Exit criteria:
- a new user can tell what backend is active and why
- local exact use is documented separately from hosted-engine use
- package installation no longer feels ambiguous

## Step 8: Produce One Decision-Grade Benchmark Pack

Why:
- broad efficiency claims are not yet justified
- one high-quality benchmark report is more valuable than many scattered runs

Required work:
- [ ] Freeze one benchmark methodology note:
  - hardware
  - workloads
  - model
  - storage format
  - candidate-generator settings
- [ ] Include exact full scan as the correctness anchor
- [ ] Include at least one native stage-1 path
- [ ] Include candidate recall, judged quality, latency, vector count, and
  byte count
- [ ] Include at least one harder public text lane and one scalable synthetic
  lane
- [ ] Add a rerun command surface that an outsider can follow

Exit criteria:
- the repo has one benchmark pack that can support decision-grade discussion
- performance claims reference that pack, not scattered traces

## Step 9: Close The Public Story Loop

Why:
- the product story is now clearer in the docs than in the main public
  surfaces

Required work:
- [ ] Update the top-level README to reflect the final category cleanly
- [ ] Update the Python README to match the SDK role
- [ ] Add one service-facing README or operator guide
- [ ] Ensure Chroma, Qdrant, and Firnflow comparisons inform the messaging
  without overclaiming

Exit criteria:
- the main public entrypoints all tell the same story
- a new reader can understand the SDK vs Engine split quickly

## Step 10: External Validation

Why:
- some `5/5` claims cannot be earned from local code alone

Required work:
- [ ] Run one clean external consumer validation for the SDK
- [ ] Run one clean external self-hosted service validation
- [ ] Run one operator-style restore or failure drill
- [ ] Collect one design-partner or outside-user feedback pass
- [ ] Fix the highest-signal issues found in those passes

Exit criteria:
- adoption readiness is supported by evidence outside the main dev checkout
- at least one outsider can use the documented flow without repo-level context

## Repo-Controlled Vs External Steps

These steps are mostly repo-controlled:
- Step 1
- Step 2
- Step 3
- Step 4
- Step 5
- Step 7
- Step 8
- Step 9

These require product decisions in addition to code:
- Step 1
- Step 4
- Step 6

These require external validation:
- Step 10

Reason:
- this keeps the plan honest
- "complete" cannot mean only "merged a patch"

## Definition Of Complete

Kayak reaches a defensible `5/5` across all current axes only when:

- the trust boundary is explicit and consistently documented
- one encoder space per collection is enforced and documented
- the typed service contract has a real deployable service path
- self-hosted deployment is documented and validated
- operator recovery and retention workflows are documented and tested
- the Python SDK has a clear install and backend story
- one decision-grade benchmark pack exists and is rerunnable
- the README and public docs tell one coherent category story
- at least one external validation pass succeeds

## Immediate Next Move

The best next move is:

1. complete Step 1 and Step 2 together
2. then build Step 3 and Step 4 as one deployable service tranche
3. then finish Step 7 and Step 8 before making stronger public claims

Reason:
- trust model and encoder boundary are foundational
- a service without a boundary story is risky
- performance claims without a decision-grade benchmark pack are premature
