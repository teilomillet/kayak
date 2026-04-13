# Trace: Richer Stage-1 Build Policy

Date: `2026-04-13`

## Why This Follow-On Exists

The previous tranche introduced a collection-scoped stage-1 build policy, but
only for config-free families.

That was useful for fixing the architectural seam, but still too narrow for the
families already present in the repository:

- `document_proxy` has a vector-budget knob
- `centroid_postings` has a centroid-budget knob
- `centroid_heads` adds `posting_cap`
- `gem_graph` requires graph-construction parameters

If those knobs had been added as new top-level collection fields, the collection
core would have immediately become family-shaped again.

## What Changed

### Generic config carrier

`SearchArtifactBuildSpec` now has:

- `family`
- `root`
- `config: List[SearchArtifactBuildConfigEntry]`

The policy remains a registry of specs. The collection manifest still does not
know what a centroid head or GEM graph is.

### Family-specific parsing moved behind the registry

The interpretation of `config` entries now lives in
`kayak/collections/search_artifact_builders.mojo`.

That file is responsible for:

- validating allowed config keys per supported family
- validating required keys and numeric constraints
- dispatching to the actual storage builders during segment sealing

### Supported configured families

Segment sealing now supports configured policies for:

- `document_proxy`
- `centroid_postings`
- `centroid_heads`
- `gem_graph`

The collection core remains generic because those families are only decoded in
the builder layer.

## What Was Verified

Focused tests:

- `tests/test_collection_contracts.mojo`
- `tests/test_collection_storage.mojo`
- `tests/test_service_contracts.mojo`
- `tests/test_service_json.mojo`
- `tests/test_segment_builder_policy.mojo`

Dependent tests:

- `tests/test_collection_resolution.mojo`
- `tests/test_snapshot_bundle.mojo`
- `tests/test_service_runtime.mojo`

Notable end-to-end confirmation:

- the hosted runtime can now create a collection whose default stage-1 policy
  builds a configured `gem_graph` sidecar at snapshot seal time and execute a
  `gem_graph` search plan against that snapshot

## What Still Remains Out Of Scope

This does not yet claim that:

- all future family configs should share the same meaning
- adaptive GEM config belongs in the current mainline default builder API
- cross-family dependencies should be expressed only through free-form key/value
  payloads forever

It only claims that the current codebase now has a reversible seam for richer
family-specific build parameters.
