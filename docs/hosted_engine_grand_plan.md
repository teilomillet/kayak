# Hosted-Engine Grand Plan

Date: `2026-04-13`

This note defines the next-step grand plan for `kayak` with two goals:

- turn the current hosted late-interaction loop into a stronger engine
- record concrete optimization suggestions file by file

This is intentionally epistemic.

The recommendations below are based on files that were actually reviewed in the
current repository. The scope is the hosted-engine critical path, not every
benchmark helper or every historical experiment file.

## Why This Plan Exists

Current repo state:

- the engine already has explicit `SearchPlan` stages
- hosted collection create, mutate, snapshot, export/import, search, and
  explain already exist
- the hosted-engine P0 mainline tranche described here is now implemented on
  the current branch
- stage-aware evaluation already reports exact-reference candidate recall
- stronger 2030 efficiency claims remain only partially established

That combination implies a clear next move:

- prioritize hosted-engine continuity and operational correctness first
- keep stage-1 architecture exploration parallel and comparable
- keep broader efficiency claims measurable, but do not make them the mainline
  decision driver yet

## Situational Awareness

### What is already strong

- explicit storage contracts in `kayak/collections/`
- explicit service contracts in `kayak/service/`
- explicit stage-aware execution in `kayak/planning/`
- exact CPU correctness anchor in `kayak/runtime/`
- a narrow stronger ceiling path in `kayak/verifier/`
- append-style hosted draft mutations
- explicit seal and publish helpers in the hosted snapshot path
- capability-aware snapshot resolution
- generic search-artifact manifests exercised by the mainline seal path
- published-state service metrics
- executable live-snapshot compaction and replacement
- file-atomic manifest and sidecar writes on the hosted path
- explicit `active_snapshot_id` publication semantics
- compacted draft baselines plus mutation replay after snapshot publication
- exact metadata filtering on the hosted exact path
- operational counters for inactive snapshots and pending draft mutations

### What is still weak

- publication is only file-atomic, not a transactional multi-file commit across
  segment sealing and snapshot promotion
- superseded snapshots and inactive segments are measured, but no reclaim or
  retention policy exists yet
- search-native sidecars still need policy-driven build selection beyond
  today's baseline registry
- hosted execution supports exact metadata filtering, but candidate-pushdown and
  approximate filter-aware stage-1 generation still do not exist
- service metrics now expose operational counters, but those counters do not
  yet drive runtime behavior
- current stronger ceiling support is still narrow and benchmark-oriented

### Resulting strategy

The next mainline work should optimize for:

1. append-friendly hosted mutations
2. repeatable snapshot publication
3. artifact-registry readiness for multiple stage-1 families
4. lazy and capability-aware snapshot resolution
5. filter and metadata readiness
6. operable metrics and explain surfaces
7. a cleaner stronger-ceiling hook

## Optimization Axes

### Axis A: Mutation And Seal Path

Problem:

- `kayak` currently seals one whole draft into one whole segment and rebuilds
  stage-1 sidecars eagerly

Why it matters:

- this is fine for correctness
- it is the wrong steady-state shape for a hosted engine

Desired outcome:

- draft ingestion becomes append-friendly
- seal is a deliberate stage boundary
- sidecar construction becomes policy-driven instead of hard-coded

### Axis B: Snapshot Publication And Lifecycle

Problem:

- snapshots exist, but collection state does not yet express a canonical active
  snapshot or a publication model

Why it matters:

- clients and operators need a stable visible snapshot boundary
- compaction and import/export need clearer promotion semantics

Desired outcome:

- draft, sealed, published, and compacted states become explicit

### Axis C: Sidecar Generalization

Problem:

- segment manifests currently have dedicated fields for each sidecar family

Why it matters:

- this will become brittle as new engine families such as GEM-style native
  paths arrive

Desired outcome:

- exact index remains first-class
- search-native sidecars become a versioned registry rather than field
  proliferation

### Axis D: Filter-Ready Hosted Search

Problem:

- typed filters exist, but hosted runtime rejects anything beyond `match_all`

Why it matters:

- multi-tenant layout and shared segment pools depend on filter-aware
  candidate generation and exact-stage pruning

Desired outcome:

- metadata sidecars and filter capability checks become first-class engine
  concerns

### Axis E: Resolver And Load Performance

Problem:

- resolved snapshots eagerly materialize artifacts and use copy-heavy
  structures

Why it matters:

- hosted search will eventually need selective artifact loading and cleaner
  memory behavior

Desired outcome:

- snapshot resolution becomes lazy, capability-aware, and explicit about what
  is loaded

### Axis F: Stronger Ceiling Boundary

Problem:

- a stronger local ceiling exists only in a narrow reranker path

Why it matters:

- the late-interaction 2030 framing requires a stronger comparison ceiling, not
  only exact MaxSim versus approximate stage 1

Desired outcome:

- stronger reranking or richer verification becomes a composable stage rather
  than a benchmark-only special case

## Reviewed File Set

The following files were reviewed for this plan:

- `docs/architecture/service_api.md`
- `docs/architecture/segment_storage.md`
- `docs/architecture/tenant_layout.md`
- `docs/late_interaction_2030.md`
- `docs/late_interaction_efficiency_roadmap.md`
- `docs/epistemic_status.md`
- `kayak/service/runtime.mojo`
- `kayak/service/collection_requests.mojo`
- `kayak/service/document_requests.mojo`
- `kayak/service/snapshot_requests.mojo`
- `kayak/service/search_contracts.mojo`
- `kayak/service/json.mojo`
- `kayak/service/draft_state.mojo`
- `kayak/service/service_status.mojo`
- `kayak/service/paths.mojo`
- `kayak/collections/collection.mojo`
- `kayak/collections/segment.mojo`
- `kayak/collections/snapshot.mojo`
- `kayak/collections/paths.mojo`
- `kayak/collections/collection_store.mojo`
- `kayak/collections/segment_store.mojo`
- `kayak/collections/snapshot_store.mojo`
- `kayak/collections/snapshot_bundle_store.mojo`
- `kayak/collections/snapshot_transfer.mojo`
- `kayak/collections/resolver.mojo`
- `kayak/collections/resolved_snapshot.mojo`
- `kayak/collections/text_corpus_store.mojo`
- `kayak/collections/report.mojo`
- `kayak/collections/report_json.mojo`
- `kayak/collections/segment_report.mojo`
- `kayak/collections/compaction.mojo`
- `kayak/storage/manifest.mojo`
- `kayak/storage/packed_index_store.mojo`
- `kayak/storage/document_proxy_store.mojo`
- `kayak/storage/centroid_postings_store.mojo`
- `kayak/storage/metadata.mojo`
- `kayak/collections/artifact_manifest.mojo`
- `kayak/planning/search_plan.mojo`
- `kayak/planning/execution.mojo`
- `kayak/planning/candidate_set.mojo`
- `kayak/planning/stage_profile.mojo`
- `kayak/planning/explain.mojo`
- `kayak/runtime/exact_cpu_backend.mojo`
- `kayak/verifier/pipeline.mojo`
- `kayak/verifier/clause_text.mojo`
- `tests/test_service_runtime.mojo`
- `tests/test_service_contracts.mojo`
- `tests/test_filters.mojo`

## File-By-File Suggestions

### Architecture And Strategy Notes

- `docs/architecture/service_api.md`: Add an explicit publication model section.
  Recommendation: distinguish `created`, `sealed`, `published`, and
  `compacted` states, and state whether a collection has an `active_snapshot`.
  Today the doc is honest that `snapshot_id` must be explicit; the next hosted
  step should make that lifecycle more concrete.

- `docs/architecture/segment_storage.md`: Expand the contract note into an
  executable artifact-registry plan. Recommendation: add a section for
  "exact artifact" versus "search-native sidecar artifacts" versus "metadata
  sidecars" so future GEM-style work does not force new dedicated manifest
  fields for every sidecar family.

- `docs/architecture/tenant_layout.md`: Add a staged rollout plan for
  filter-aware serving. Recommendation: define the exact milestone at which
  tenant-isolated roots remain mandatory, and the later milestone at which
  shared physical pools become sound.

- `docs/late_interaction_2030.md`: Add one sentence linking hosted-engine
  continuity to the abstract local-interaction-plus-sublinear-search framing.
  Recommendation: make explicit that the service/storage loop is the engine
  manifestation of that paradigm, not a separate product track.

- `docs/late_interaction_efficiency_roadmap.md`: Add a dependency note that
  Track B should plug into the hosted engine rather than bypass it. This will
  keep GEM or future native work measured through the same stage-aware service
  surface.

- `docs/epistemic_status.md`: Keep this file updated whenever a broad claim
  changes status. Recommendation: treat it as the canonical "claim ledger" and
  require new traces to point back into it.

### Service Boundary

- `kayak/service/runtime.mojo`: This is the highest-priority file to optimize.
  Recommendation:
  - split `create_snapshot(...)` into explicit helpers such as
    `seal_draft_to_segment`, `build_sidecars_for_segment`, and
    `publish_snapshot`
  - stop treating snapshot creation as a monolithic whole-draft rewrite forever
  - replace `require_filter_is_match_all(...)` with a capability check layer so
    filters can progress from `rejected` to `exact_only` to `candidate_pushdown`
  - add real metric updates for create, mutate, snapshot, search, explain, and
    import/export operations
  Reason:
  - the current runtime is correct but still scaffold-shaped

- `kayak/service/collection_requests.mojo`: Add collection policy fields that
  the engine will actually need. Recommendation:
  - snapshot publication policy
  - default sidecar build policy
  - metadata schema version or contract marker
  This prevents those choices from leaking into ad hoc runtime defaults later.

- `kayak/service/document_requests.mojo`: Extend the mutation contract beyond
  vectors plus optional text. Recommendation:
  - add optional metadata payloads
  - add explicit mutation semantics such as `upsert`, `replace`, or `delete`
    policy if the service later needs idempotent writes
  Reason:
  - filters and tenant-aware serving will eventually require metadata at ingest

- `kayak/service/snapshot_requests.mojo`: Add publication semantics.
  Recommendation:
  - support an explicit "publish after create" or "create only" flag
  - reserve room for labels or retention reason codes
  Reason:
  - snapshots are currently created and stored, but not yet fully lifecycle-aware

- `kayak/service/search_contracts.mojo`: Keep typed plans, but add a thinner
  preset layer. Recommendation:
  - keep `SearchPlan` explicit internally
  - add room for service presets or plan aliases so callers do not have to
    construct generator-specific plans for common paths
  Also add timeout or execution-hint fields once the service runtime becomes
  more operational.

- `kayak/service/json.mojo`: Replace hand-written JSON assembly with a shared
  encoding helper or generated codec approach once the contract stabilizes.
  Recommendation:
  - centralize escaping and field emission
  - add schema version markers
  - avoid letting this file become a second source of truth for the service API

- `kayak/service/draft_state.mojo`: This is the mutation-path bottleneck.
  Recommendation:
  - stop rewriting the entire draft state as one packed index plus one text
    corpus after every mutation
  - move toward append-oriented draft batches or a tiny WAL-like layer
  - keep deletes as tombstones until seal time
  Reason:
  - the current shape is simple, but it will scale poorly for hosted usage

- `kayak/service/service_status.mojo`: Turn the contracts into something the
  runtime can actually populate. Recommendation:
  - add counters for draft documents, published snapshots, pending segment
    builds, compaction backlog, and optionally last snapshot generation per
    collection

- `kayak/service/paths.mojo`: Add staging and lifecycle roots.
  Recommendation:
  - introduce path helpers for temporary segment builds, pending compaction
    outputs, and possibly published-snapshot pointers
  Reason:
  - current paths handle draft and collections only, but not staged lifecycle
    transitions

### Collection And Snapshot Model

- `kayak/collections/collection.mojo`: Add explicit publication metadata.
  Recommendation:
  - consider `active_snapshot_id` or a similar pointer once the lifecycle
    semantics are decided
  - reserve room for collection policy versioning
  Reason:
  - `latest_generation` alone is not the same thing as "currently published"

- `kayak/collections/segment.mojo`: Reduce field explosion.
  Recommendation:
  - move from dedicated optional roots
    (`centroid_postings_root`, `centroid_heads_root`, `document_proxy_root`)
    toward a generic artifact inventory or sidecar registry
  Reason:
  - GEM and future native engines are orthogonal to current centroid families
  - this file is where schema brittleness will show first

- `kayak/collections/snapshot.mojo`: Add snapshot role semantics.
  Recommendation:
  - support a clear distinction between retained snapshots and the currently
    published one
  - leave room for retention metadata

- `kayak/collections/paths.mojo`: Add helpers for artifact-registry roots and
  publication pointers. Recommendation:
  - avoid scattering path conventions across runtime and transfer code

- `kayak/collections/collection_store.mojo`: Add safer manifest updates.
  Recommendation:
  - use atomic write patterns
  - enforce monotonic generation updates
  - centralize any future active-snapshot update here instead of in service
    runtime

- `kayak/collections/segment_store.mojo`: Generalize artifact-root encoding.
  Recommendation:
  - replace the repeated encode/decode helper pattern with a common optional
    artifact-root codec or a structured sidecar list
  Reason:
  - this is the exact persistence seam where new sidecar families will multiply

- `kayak/collections/snapshot_store.mojo`: Add atomic publish helpers.
  Recommendation:
  - keep simple manifest IO, but add helpers for publish/promote operations
    once active snapshots exist
  - eventually treat snapshot creation and publication as distinct actions

- `kayak/collections/snapshot_bundle_store.mojo`: Add integrity support.
  Recommendation:
  - store checksums or content fingerprints for imported/exported bundles
  - this matters once snapshots move beyond local smoke use

- `kayak/collections/snapshot_transfer.mojo`: Separate compatibility checks
  from copy mechanics. Recommendation:
  - factor import compatibility logic into reusable validators
  - add checksum verification and collision policy
  - treat transfer as a service-facing lifecycle primitive, not just a file copy

- `kayak/collections/compaction.mojo`: This file is currently only a data
  contract. Recommendation:
  - add a real compaction executor and a merge policy note
  - make compaction publish-safe by writing replacement segments before any
    snapshot promotion
  Reason:
  - compaction is one of the main remaining hosted-engine gaps

### Resolver, Reports, And Load Path

- `kayak/collections/resolver.mojo`: This is the highest-priority read-path
  file after service runtime.
  Recommendation:
  - add lazy resolution modes such as:
    - exact-only
    - exact plus text
    - exact plus stage-1 sidecar family
  - avoid loading artifacts that the current plan cannot use
  - consider handle-based or borrowed representations instead of always copying
  Reason:
  - the current resolver is robust but eager

- `kayak/collections/resolved_snapshot.mojo`: Make loaded segments
  capability-aware.
  Recommendation:
  - add a representation that separates manifest data from loaded artifact
    payloads
  - this will make lazy loading and future caching easier

- `kayak/collections/report.mojo`: Add deeper operational reporting.
  Recommendation:
  - report artifact-family byte breakdown
  - add stage-1 sidecar presence counts
  - report snapshot-level density, not only aggregate counts

- `kayak/collections/report_json.mojo`: Keep it aligned with the richer report.
  Recommendation:
  - include per-artifact byte breakdown and lifecycle state once the report has
    it

- `kayak/collections/segment_report.mojo`: Extend beyond text/no-text.
  Recommendation:
  - include which sidecars are present
  - include exact-index bytes versus sidecar bytes

- `kayak/collections/mirror.mojo`: Reclassify this helper as a fixture-oriented
  path instead of a production collections primitive.
  Recommendation:
  - keep it, but document it as benchmark support
  - avoid letting "mirror" semantics steer the hosted-engine design

- `kayak/collections/text_corpus_store.mojo`: Optimize the baseline text codec.
  Recommendation:
  - keep file-per-doc UTF-8 as the exact-preservation baseline
  - add a packed blob plus offsets alternative for larger hosted collections
  - add integrity checks that align stored doc ids and payload offsets

### Storage Artifacts

- `kayak/storage/manifest.mojo`: Add atomicity and migration support.
  Recommendation:
  - write manifests atomically
  - add room for checksum or schema migration helpers
  - keep format negotiation centralized here

- `kayak/storage/packed_index_store.mojo`: This file needs both correctness and
  performance tightening.
  Recommendation:
  - stop returning `VECTOR_SCALAR_NAME` unconditionally on load; use the stored
    manifest value after validation
  - add artifact byte size to the manifest for parity with sidecars
  - prepare for lazy or memory-mapped loading later

- `kayak/storage/document_proxy_store.mojo`: Reduce manifest rewrite
  duplication.
  Recommendation:
  - factor the two-pass `artifact_byte_size` manifest write into a shared helper
  - add integrity checks or checksums
  - keep this store aligned with any future generic sidecar registry

- `kayak/storage/centroid_postings_store.mojo`: Treat this as a generic
  search-native artifact store, not only a centroid family store.
  Recommendation:
  - factor common artifact-byte-size stabilization helpers
  - make block summaries and posting-order variants clearer in manifest terms
  - prepare the manifest shape so GEM-style artifacts can reuse similar IO
    discipline without cloning this whole file

- `kayak/storage/metadata.mojo`: Reconsider the storage structs as the number
  of sidecar types grows.
  Recommendation:
  - keep explicit structs for exact and current sidecars
  - but plan for either:
    - a shared artifact-header type
    - or a sidecar capability trait
  Reason:
  - this is the type-level pressure point for future engine families

- `kayak/collections/artifact_manifest.mojo`: Make this the canonical place for
  collection artifact versioning and vector-scalar checks. Recommendation:
  - keep its responsibility tight
  - add room for artifact checksum fields here before duplicating them
    elsewhere

### Planning And Execution

- `kayak/planning/search_plan.mojo`: Separate plan contract from convenience
  constructors.
  Recommendation:
  - keep `SearchPlan` small and stable
  - move generator-specific convenience constructors or preset bundles into a
    dedicated preset layer once the list gets longer
  Reason:
  - this file will otherwise keep expanding as more stage-1 families arrive

- `kayak/planning/execution.mojo`: Replace the large generator `if` chain with
  a capability-oriented dispatch boundary.
  Recommendation:
  - introduce a stage-1 execution interface or dispatch table
  - keep segment iteration and candidate accounting shared
  - make generator-specific logic pluggable
  Reason:
  - this is the main code hotspot where centroid families and GEM-style paths
    will otherwise collide

- `kayak/planning/candidate_set.mojo`: Add a little more operational context.
  Recommendation:
  - keep the current shape
  - add optional fields later for filter selectivity, loaded artifact bytes, or
    candidate-stage wall time

- `kayak/planning/stage_profile.mojo`: Add time and selectivity fields.
  Recommendation:
  - include measured wall time per stage
  - include filter selectivity once filters are supported
  Reason:
  - current explain outputs are strong on counts, weaker on runtime timing

- `kayak/planning/explain.mojo`: Preserve exact-reference explain, but make it
  extensible.
  Recommendation:
  - add a place for service-level timing and filter information
  - keep faithfulness assessment explicit
  - ensure stronger-ceiling stages can be attached without inventing a second
    explain object

### Runtime And Ceiling Path

- `kayak/runtime/exact_cpu_backend.mojo`: Expose more backend control.
  Recommendation:
  - keep it minimal for correctness
  - add optional knobs for profiling, thread/affinity control, or execution
    mode once service benchmarking becomes more operational

- `kayak/verifier/pipeline.mojo`: Turn verifier use into a true stage contract.
  Recommendation:
  - integrate it more directly with `SearchPlan` or a future reranker stage
    rather than keeping it as a separate exact-search helper path
  Reason:
  - stronger local ceilings should become part of the engine story, not only
    offline experiments

- `kayak/verifier/clause_text.mojo`: Keep this as a narrow baseline, but do not
  let it become the final ceiling implementation.
  Recommendation:
  - factor reusable text-reranker interfaces out of it
  - preserve this heuristic path as one local ceiling baseline
  - prepare for stronger rerankers later without rewriting benchmark logic

### Tests

- `tests/test_service_runtime.mojo`: Expand beyond the happy path.
  Recommendation:
  - add multi-snapshot publication cases
  - add import conflict cases
  - add failure cases for unsupported filters
  - add sidecar-presence assertions after sealing

- `tests/test_service_contracts.mojo`: Add lifecycle and policy coverage.
  Recommendation:
  - test future active-snapshot or publication semantics once added
  - test search preset translation if a service preset layer is introduced

- `tests/test_filters.mojo`: Keep the grammar tests, but add execution-facing
  cases later.
  Recommendation:
  - extend once metadata sidecars and filter pushdown exist
  - do not let filter support appear complete while runtime still rejects it

## Suggested New Files

These do not need to be written immediately, but the current design pressure
suggests them:

- `kayak/collections/segment_builder.mojo`
  - isolate sealing and sidecar-build policy from service runtime
- `kayak/collections/publish.mojo`
  - isolate active-snapshot promotion semantics
- `kayak/collections/metadata_store.mojo`
  - persist segment-local document metadata for filters
- `kayak/planning/stage1_registry.mojo`
  - isolate generator dispatch from `execution.mojo`
- `kayak/service/metrics_runtime.mojo`
  - keep service counters and aggregation out of the main runtime file
- `docs/hosted_engine_lifecycle.md`
  - document ingest, seal, publish, compact, import/export states explicitly

## Ordered Execution Plan

### P0: Mainline Must Do

1. Refactor `kayak/service/runtime.mojo` around explicit seal and publish steps.
2. Refactor `kayak/service/draft_state.mojo` away from full-state rewrites.
3. Refactor `kayak/collections/resolver.mojo` toward lazy capability-aware
   loading.
4. Refactor `kayak/planning/execution.mojo` toward pluggable stage-1 dispatch.
5. Extend `kayak/collections/segment.mojo` and `segment_store.mojo` toward a
   generic sidecar registry.

### P1: Mainline Soon After

1. Add metadata sidecars and staged filter support.
2. Add real compaction execution and publication-safe replacement.
3. Add stronger service metrics and richer storage/report JSON.
4. Add lifecycle-heavy hosted runtime tests.

### P2: Parallel But Not Mainline-Blocking

1. Let GEM-style stage-1 work plug into the generalized sidecar and
   dispatch boundaries.
2. Grow the stronger local ceiling boundary beyond `clause_text`.
3. Revisit broad efficiency claims only after the hosted engine and sidecar
   model are more stable.

## Abstract Vision

The correct abstract vision for `kayak` is not:

- a benchmark harness that happens to have storage
- or a vector database with a late-interaction plugin

The correct vision is:

- a snapshot-based hosted engine for late interaction
- where exact late interaction is the correctness anchor
- where stage 1 is a pluggable segment-local acceleration layer
- where sidecars are first-class sealed artifacts
- where filters, explainability, and publication semantics are engine
  primitives

In that vision:

- a document enters a draft mutation path
- draft state is sealed into immutable segments
- sidecars are built per segment according to policy
- a snapshot publishes the visible search state
- search resolves only the artifacts required by the chosen plan
- exact late interaction reranks against the published snapshot
- stronger ceilings remain explicit and measurable instead of hidden
- compaction rewrites old segments without mutating the visible state in place

That is the shape that keeps `kayak` useful even if:

- centroid families are not the final stage-1 winner
- GEM-style native engines become the preferred primitive
- stronger compression paths change storage assumptions
- stronger ceiling paths change what "good enough" means

This is why hosted-engine continuity is the highest-confidence next focus.
