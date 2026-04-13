# Tenant Layout Architecture

Status: `Phase B design note`  
Date: `2026-04-12`

This document explains how `kayak` should think about per-tenant and
cross-tenant physical layout for late-interaction collections.

It is intentionally epistemic:
- verified statements are tied to the current repository
- source-backed inferences are labeled
- design choices that are not yet implemented remain explicitly provisional

## Verified Starting Point

These statements are checked against the current repository.

1. `CollectionManifest`, `SealedSegmentManifest`, and `SnapshotManifest` all
   carry explicit `tenant_id` and `namespace_id`.
2. The current collection resolver rejects mismatches across collection,
   snapshot, and segment tenant boundaries.
3. The current physical model is one tenant-scoped collection root at a time.
4. Snapshot export/import now preserves tenant and namespace identity inside the
   exported bundle.
5. The repository now has:
   - persisted document metadata sidecars for arbitrary document fields
   - native filter-aware candidate generation for exact `doc_id` and metadata
     filters
   - internal logical-scope postings in `document_filter_index` for new
     segments, so filtered search can carry collection, tenant, and namespace
     scope without exposing those fields publicly
6. The repository still does not have:
   - shared physical segment pools across multiple tenants
   - explicit `match_all` shared-pool scope pushdown independent of the current
     tenant-rooted collection layout

## Source-Backed Inference

Curator is the main external evidence for this design area:
- https://arxiv.org/abs/2401.07119

Inference from that baseline:
- multi-tenant retrieval is not just an access-control problem
- layout choices affect latency, memory efficiency, and update behavior
- an engine should avoid forcing users into only two extremes:
  one giant shared index or one tiny isolated index for every tenant

## Decision

Treat tenant isolation as a logical invariant and physical layout as a policy.

That means:
- every collection, segment, snapshot, and bundle keeps explicit tenant and
  namespace identifiers
- physical placement may evolve later
- the engine must never rely on an implicit caller-side filter to restore
  tenant isolation after the fact

## Current Default Recommendation

Use tenant-isolated collection roots as the default serving layout.

Why this is the current default:
- it matches the current codebase
- it is easy to reason about operationally
- it keeps snapshot export/import simple
- it avoids accidental cross-tenant leakage while filter and metadata layers are
  still immature

This default is especially appropriate for:
- small tenants
- fast-moving tenants with frequent updates
- early service deployments where correctness is more important than maximal
  packing density

## Future Shared-Layout Recommendation

Cross-tenant physical sharing should be considered only after filter pushdown
and metadata storage exist.

The plausible future shape is:
- keep tenant identity explicit per segment and per posting domain
- admit physically shared segment pools only when stage-1 candidate generation
  can enforce tenant and metadata constraints before expensive late interaction
- preserve snapshot reproducibility even when segments are shared

Inference:
- a shared physical pool may help large fleets of tiny tenants
- but the quality and safety boundary still depends on filter-aware candidate
  generation, not on path naming alone

## Two Layout Families To Support

`kayak` should plan for two physical layout families:

1. Tenant-isolated collections
   - one tenant and namespace per collection root
   - sealed segments belong to exactly one tenant
   - the simplest and safest default

2. Shared segment pools
   - multiple tenants can reference shared physical segment groups
   - only valid once filter-aware stage-1 retrieval exists
   - must preserve explicit tenant accounting in manifests and metrics

## What The Filter Model Enables

The repository now has a canonical typed filter-expression model for search
requests under [`kayak/filters/`](../../kayak/filters).

That does not mean filter execution is finished.

What it enables now:
- a stable request grammar
- low-selectivity and high-selectivity benchmark fixtures
- future candidate-generation work that can reason about selectivity as an
  engine concern
- filtered hosted search can now attach internal collection, tenant, and
  namespace scope to the candidate-generation filter path when a segment carries
  scope-aware `document_filter_index` postings

What it does not yet enable:
- unfiltered shared-index serving without further work
- a public caller-visible scope-filter vocabulary

## Immediate Follow-On Work

The next tenant-layout work should be:

1. benchmark low-selectivity versus high-selectivity workloads explicitly on the
   scope-aware filter sidecar path
2. extend logical-scope pushdown from filtered requests to explicit shared-pool
   `match_all` serving
3. only then prototype shared physical segment pools

That order keeps isolation sound while opening a path to higher density later.
