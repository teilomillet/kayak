# Distributed Sharding Defer Note

Status: `explicitly deferred`  
Date: `2026-04-12`

This note records a deliberate non-decision:
`kayak` should not design distributed sharding yet.

## Why This Is The Sound Choice

The repository now has:
- collection manifests
- sealed segment manifests
- snapshot/export/import boundaries
- tenant and namespace layout notes
- filter-expression contracts
- a typed service boundary

Those are necessary prerequisites, but they are not yet sufficient for a sound
distributed design.

The missing stabilizers are still:
- production ingest and compaction behavior across multiple generations
- filter-aware candidate generation beyond exact full scan
- explicit active-snapshot and routing semantics for hosted collections
- measured CPU/GPU execution-stage boundaries under real serving loads

Inference:
- designing shard routing now would force assumptions about segment placement,
  replica ownership, and cross-shard top-`k` merge behavior before the local
  single-node engine contracts have settled
- that would create more speculative architecture than usable infrastructure

## Current Rule

Until those prerequisites settle:
- keep search execution single-node and snapshot-scoped
- keep tenant, namespace, collection, segment, and snapshot identities explicit
- do not introduce cross-node query routing into the service boundary
- do not hide future sharding assumptions inside collection manifests

## Revisit Trigger

Distributed sharding becomes a live design topic only after all of the
following are true:
- the service boundary is exercised through a thin HTTP/JSON adapter
- compaction and snapshot behavior are stable for hosted collections
- candidate generation has at least one non-exact stage with measurable recall
  tradeoffs
- runtime boundaries for CPU and future GPU exact scoring are stable

That is the current defer decision.
