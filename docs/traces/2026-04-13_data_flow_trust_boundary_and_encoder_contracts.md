# 2026-04-13 Data Flow, Trust Boundary, And Encoder Contracts

## Goal

Make Kayak's external data flow more explicit and more honest by:

- defining what enters and leaves the system
- defining the default trust boundary
- defining what "one encoder space per collection" means in product terms
- adding service-level validation where the existing surface allowed ambiguity

## Why This Tranche

Before this change, the repo already had:
- explicit collection and snapshot manifests with `model_name`,
  `vector_scalar_name`, and `vector_dim`
- resolver and import checks that rejected incompatible stored artifacts
- a service contract that treated the encoder boundary as external

But it still lacked two things:
- one direct architecture note explaining how data is consumed, digested, and
  returned
- one explicit hosted search field for the caller's claimed query model space

That left a gap:
- a same-dimension query from the wrong model family could be treated as
  shape-compatible even though it was semantically unsafe

## Changes

Added docs:
- [docs/architecture/data_flow_io.md](../architecture/data_flow_io.md)
- [docs/architecture/trust_boundary.md](../architecture/trust_boundary.md)
- [docs/architecture/multi_encoder_interoperability.md](../architecture/multi_encoder_interoperability.md)

Updated docs:
- [docs/architecture/service_api.md](../architecture/service_api.md)
- [docs/product_positioning.md](../product_positioning.md)
- [README.md](../../README.md)
- [TODO.md](../../TODO.md)

Updated service contract and runtime:
- [kayak/service/search_contracts.mojo](../../kayak/service/search_contracts.mojo)
- [kayak/service/runtime.mojo](../../kayak/service/runtime.mojo)
- [kayak/service/json.mojo](../../kayak/service/json.mojo)

Updated tests and example:
- [tests/test_service_contracts.mojo](../../tests/test_service_contracts.mojo)
- [tests/test_service_runtime.mojo](../../tests/test_service_runtime.mojo)
- [tests/test_snapshot_bundle.mojo](../../tests/test_snapshot_bundle.mojo)
- [tests/test_service_json.mojo](../../tests/test_service_json.mojo)
- [tests/test_planned_stage2_override.mojo](../../tests/test_planned_stage2_override.mojo)
- [examples/hosted_collection_smoke.mojo](../../examples/hosted_collection_smoke.mojo)

## Design Decisions

### 1. Query model identity is explicit at the service boundary

Decision:
- `SearchRequest` and `PlannedSearchRequest` now carry `query_model_name`

Reason:
- collection manifests already define one encoder space
- the service should reject wrong-model queries before scoring instead of only
  relying on shape compatibility

### 2. Canonical input stays late-interaction-native

Decision:
- the new data-flow note recommends encoded late-interaction vectors or
  Kayak-native packed artifacts as the canonical external boundary
- LanceDB or similar systems remain adapter candidates, not the source of
  truth for Kayak's meaning of a document or query

Reason:
- Kayak's differentiator is explicit late-interaction semantics, not generic
  vector-row storage

### 3. Multi-encoder support remains a non-goal by default

Decision:
- arbitrary heterogeneous late interaction is documented as unsupported by
  default

Reason:
- the repo already enforces one model space in manifests and imports
- the product surface should match that reality instead of implying unsafe
  flexibility

## Verification Commands

```bash
pixi run mojo -I . tests/test_service_contracts.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . tests/test_snapshot_bundle.mojo
pixi run mojo -I . tests/test_service_json.mojo
pixi run mojo -I . tests/test_planned_stage2_override.mojo
pixi run mojo -I . examples/hosted_collection_smoke.mojo
git diff --check
```

## Verified Results

- `tests/test_service_contracts.mojo`
  - `24/24` passed
- `tests/test_service_runtime.mojo`
  - `25/25` passed
  - includes new rejection tests for:
    - wrong upsert vector dimension
    - wrong query model name
    - wrong query vector dimension
- `tests/test_snapshot_bundle.mojo`
  - `5/5` passed
  - includes new rejection tests for:
    - imported collection model mismatch
    - imported collection vector-dimension mismatch
- `tests/test_service_json.mojo`
  - `9/9` passed
  - now checks `query_model_name` in request JSON
- `tests/test_planned_stage2_override.mojo`
  - `1/1` passed
- `examples/hosted_collection_smoke.mojo`
  - ran successfully and emitted debug JSON through the hosted collection path
- `git diff --check`
  - passed

## What This Does Not Claim

This tranche does not claim:
- that Kayak now has a finished networked HTTP or gRPC service
- that cross-model late interaction is impossible in principle
- that Kayak's deployment trust story is complete

It does establish:
- the data flow is now documented explicitly
- the default trust boundary is now documented explicitly
- the service contract now rejects wrong-model queries more honestly
