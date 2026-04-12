# Segment Storage Architecture

Status: `Phase A` contract draft  
Date: `2026-04-12`

This document explains the storage boundary that `kayak` should use for hosted
late-interaction collections.

It is intentionally epistemic:
- verified facts are tied to the current codebase
- source-backed inferences are labeled as such
- open choices are listed instead of being silently fixed in code

## Verified Starting Point

These statements are checked against the current repository.

1. `kayak/storage/` currently mixes two concerns:
   - reusable artifact codecs such as [`manifest.mojo`](../../kayak/storage/manifest.mojo),
     [`packed_index_store.mojo`](../../kayak/storage/packed_index_store.mojo), and
     [`hybrid_flat_dim128_store.mojo`](../../kayak/storage/hybrid_flat_dim128_store.mojo)
   - benchmark and public-slice caches such as
     [`scifact_cache.mojo`](../../kayak/storage/scifact_cache.mojo) and
     [`browsecomp_plus_cache.mojo`](../../kayak/storage/browsecomp_plus_cache.mojo)
2. The persisted storage model today is benchmark-oriented:
   - [`judged_task_store.mojo`](../../kayak/storage/judged_task_store.mojo) stores qrels,
     evaluation metadata, and benchmark descriptions
   - [`metadata.mojo`](../../kayak/storage/metadata.mojo) defines
     `StoredJudgedTask`, `StoredPackedIndex`, and `StoredHybridFlatDim128Index`,
     but no hosted collection, segment, snapshot, or compaction objects
3. The current binary storage approach is already justified by measurement:
   - [`docs/traces/2026-04-12_storage_v2.md`](../traces/2026-04-12_storage_v2.md)
     shows that binary vector payloads materially reduced bytes and warm-load cost
4. Scalar choices are already centralized in
   [`kayak/numeric/scalars.mojo`](../../kayak/numeric/scalars.mojo), which matches the
   repo rule that dtype choices should not be scattered.
5. An in-memory text sidecar already exists as
   [`kayak/text/document_text_corpus.mojo`](../../kayak/text/document_text_corpus.mojo),
   but there is no first-class persisted generic text-corpus artifact yet.

## Source-Backed Inferences

These are design inferences, not direct local measurements.

1. ColBERTv2 and later engine papers treat storage and execution layout as core
   problems, not bookkeeping details.
   Source baseline:
   - ColBERTv2: https://arxiv.org/abs/2112.01488
   - PLAID: https://arxiv.org/abs/2205.09707
   - WARP: https://arxiv.org/abs/2501.17788
   - GEM: https://arxiv.org/abs/2603.20336
2. Multi-tenant retrieval needs explicit storage and layout contracts rather than
   payload filters bolted on at the edge.
   Source baseline:
   - Curator: https://arxiv.org/abs/2401.07119
3. Hosted search products such as Mixedbread expose ingestion, storage, search,
   metadata, and reranking as one product surface.
   Public product evidence:
   - https://www.mixedbread.com/docs/stores/search/rerank
   - https://www.mixedbread.com/pricing

## Problem Statement

The current repo is strong at:
- exact MaxSim execution
- packed multi-vector document storage
- benchmark/task persistence
- smoke-oriented public benchmark slices

The current repo is not yet explicit about the hosted serving boundary:
- what a collection is
- what a sealed search segment is
- how a snapshot names the searchable state
- where optional text sidecars belong
- how compaction is represented

If we extend `kayak/storage/` directly with all of those concepts right now,
we will mix:
- benchmark caches
- low-level codecs
- hosted serving contracts

That would make the code harder to browse and harder to evolve.

## Decision

Create a new top-level package, [`kayak/collections/`](../../kayak/collections),
for serving-oriented storage contracts.

This is a deliberate split:
- `kayak/storage/` keeps owning persisted artifact codecs and benchmark caches
- `kayak/collections/` owns the hosted collection model

Why this is the most sound split:
- it matches the current codebase reality instead of pretending benchmark task
  storage is already a serving system
- it keeps one concept per package boundary
- it leaves room to reuse existing packed-index codecs inside a cleaner hosted
  collection model

## Phase A Contract Surface

Phase A should define contracts first, not hide half-built implementation behind
large modules.

The minimum contract set is:
- `TenantId`
- `NamespaceId`
- `CollectionId`
- `SegmentId`
- `SnapshotId`
- `SegmentStats`
- `CollectionStats`
- `CollectionManifest`
- `SealedSegmentManifest`
- `SnapshotManifest`
- `CompactionPlan`
- `StoredDocumentTextCorpus`

What this means:
- `CollectionManifest` describes a hosted collection's stable identity and vector contract
- `SealedSegmentManifest` describes one immutable search segment
- `SnapshotManifest` names the set of sealed segments that are searchable together
- `CompactionPlan` names a future rewrite of multiple sealed segments into one replacement segment
- `StoredDocumentTextCorpus` makes the optional text sidecar explicit without pretending every collection must carry raw text

## Why Sealed Segments Come First

This is a design decision backed by the roadmap and current engine shape.

Reason:
- exact search, packed indexes, and profile tooling are easiest to reason about
  when search sees immutable payloads
- mutability belongs in ingest buffers, flush logic, and snapshot updates, not in
  the hot search path

Therefore:
- search should operate over sealed segments
- snapshots should name search-visible segments
- compaction should create new sealed segments instead of rewriting a live one in place

## Logical Storage Shape

This is a proposed logical layout, not a finalized path contract.

The important contract is the object model, not the exact directory spelling.

```text
<collection-root>/
  collection.manifest.tsv
  snapshots/
    <snapshot-id>/
      manifest.tsv
      segment_ids.tsv
  segments/
    <segment-id>/
      manifest.tsv
      packed_index/
        manifest.tsv
        doc_ids.tsv
        doc_offsets.tsv
        token_vectors.bin
      text_corpus/           # optional
        manifest.tsv
        entries.tsv
        texts/
          0.txt
          1.txt
```

Why this layout is plausible:
- it extends the repo's existing manifest-plus-binary-payload pattern
- it keeps search-hot vectors in their own artifact root
- it makes the text sidecar explicitly optional
- it preserves exact UTF-8 text in the baseline codec instead of normalizing it into one-line TSV payloads

## Required Invariants

These are the core storage invariants that should hold across the service.

1. A collection has one vector contract:
   - one `vector_dim`
   - one `vector_scalar_name`
   - one model identity at the collection boundary
2. Every sealed segment must satisfy that same vector contract.
3. Search operates on snapshots, not on arbitrary half-built segment directories.
4. Every segment and collection manifest must keep vector counts explicit.
5. Text sidecars are optional and versioned separately from vector payloads.
6. Compaction never mutates the source searchable segment in place; it creates a
   replacement output that a later snapshot can adopt.

## What This Step Does Not Decide Yet

These remain open on purpose:
- exact HTTP transport schema
- candidate-generation manifests and `SearchPlan` persistence
- mutable ingest-buffer format
- filter-expression encoding
- tenant-to-path layout policy
- distributed placement and sharding policy
- default compression scheme beyond the current binary payload baseline

Those choices should follow the collection and segment contracts, not precede them.

## Code Boundary Added In This Step

This step introduces the `kayak/collections/` package as a contract layer only.

It does not yet add:
- segment manifest readers and writers
- collection loaders
- snapshot resolution logic
- compaction executors
- text-sidecar persistence codecs

That is deliberate.

The repo should first have:
- explicit names
- explicit invariants
- small files that are easy to read

before it grows another storage implementation surface.

## Immediate Next Steps

After this contract step, the next implementation work should be:

1. Add manifest codecs for `CollectionManifest`, `SealedSegmentManifest`, and `SnapshotManifest`.
2. Add a persisted `StoredDocumentTextCorpus` artifact under the same manifest discipline as packed indexes.
3. Add a loader that resolves one snapshot into search-ready sealed segments.
4. Add collection-level storage reporting:
   - document count
   - token count
   - vector count
   - byte size
5. Add service-oriented tests for:
   - snapshot visibility
   - compaction replacement
   - text-sidecar presence or absence

That sequence preserves the current repo strengths while moving toward a hosted
late-interaction engine instead of a benchmark-only scaffold.
