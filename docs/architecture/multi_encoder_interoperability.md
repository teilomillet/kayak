# Multi-Encoder Interoperability

Status: current non-goals and decision note  
Date: `2026-04-13`

This note defines what Kayak currently supports and does not support when
multiple encoder families or model spaces are involved.

It exists to prevent a specific product mistake:

- implying that arbitrary late-interaction vectors from different models can be
  safely mixed just because their dimensions happen to match

Related notes:
- [data_flow_io.md](data_flow_io.md)
- [service_api.md](service_api.md)
- [../late_interaction_2030.md](../late_interaction_2030.md)
- [../epistemic_status.md](../epistemic_status.md)

## Core Decision

One collection equals one encoder space.

In current Kayak terms, that means one collection has one:
- `model_name`
- `vector_scalar_name`
- `vector_dim`

Reason:
- exact and approximate late interaction both assume a shared similarity space
- same-dimension vectors from different encoders are not enough to justify
  shared MaxSim semantics
- product trust is better served by explicit rejection than by ambiguous
  "maybe it works" behavior

## What Is Supported

Supported today:

- one model family per collection
- one query model space per hosted request
- separate collections for separate encoders
- separate snapshots for the same collection model space over time
- import of snapshots only when the target collection manifest is compatible

Practical pattern:
- if a user wants to search multiple models, create multiple collections and
  merge or rerank results outside Kayak

## What Is Unsupported By Default

Unsupported by default:

- mixing arbitrary encoder families inside one collection
- importing a snapshot from model A into an existing collection for model B
- issuing a hosted query encoded by a different model than the collection's
  declared model space
- assuming that same `vector_dim` implies semantic compatibility

This is a product non-goal for now, not just an undocumented limitation.

## Current Enforcement Points

The repository already enforces most of this mechanically.

### Collection boundary

Collections persist:
- `model_name`
- `vector_scalar_name`
- `vector_dim`

That is stored in:
- [collection.mojo](../../kayak/collections/collection.mojo)

### Segment boundary

Sealed segments persist the same model-space contract.

That is stored in:
- [segment.mojo](../../kayak/collections/segment.mojo)

### Resolver boundary

Resolved snapshots reject segment and artifact mismatches against the declared
collection model space.

That is enforced in:
- [resolver.mojo](../../kayak/collections/resolver.mojo)

### Import boundary

Snapshot import rejects incompatible collection or segment manifests.

That is enforced in:
- [snapshot_transfer.mojo](../../kayak/collections/snapshot_transfer.mojo)

### Search boundary

Hosted search requests now carry explicit `query_model_name`.

The runtime rejects:
- wrong `query_model_name`
- wrong query `vector_dim`

That is enforced in:
- [search_contracts.mojo](../../kayak/service/search_contracts.mojo)
- [runtime.mojo](../../kayak/service/runtime.mojo)

## Why Query Model Identity Matters

Without an explicit `query_model_name`, the hosted service can only trust:
- collection identity
- query vector dimension

That is not enough.

A same-dimension query from another encoder family could still be semantically
wrong while looking shape-compatible.

Decision:
- the service boundary should carry the caller's claimed query model identity
- the runtime should reject mismatches before scoring

## What To Do If A User Wants Multiple Models

There are three honest options.

### 1. Separate collections

Use one collection per model.

Then:
- search each collection independently
- merge or rerank results outside Kayak

This is the default recommended path.

### 2. Explicit offline calibration or research workflow

If a team has:
- joint training
- a verified calibration layer
- or a known interoperable representation trick

then that should be introduced as an explicit new engine contract, not as an
implicit use of today's search path.

### 3. Future federated search layer

A future Kayak layer could:
- fan out one request to multiple collections
- return per-collection results
- apply an explicit fusion policy

That would still be different from:
- pretending all vectors share one MaxSim space

## What This Note Does Not Claim

This note does not claim:
- that cross-model late interaction is impossible in principle
- that future calibration or interoperability work is not worthwhile
- that separate collections are the final product shape for every use case

It only defines the current product truth:
- arbitrary heterogeneous late interaction is unsupported by default
- Kayak should reject unsafe mixing instead of leaving it ambiguous

## Exit Criteria For Reconsidering This Decision

Kayak should only soften this rule if there is:

- an explicit interoperability design
- a measurable validation story
- a clear new contract for how compatibility is declared and enforced

Until then:
- one collection means one encoder space
