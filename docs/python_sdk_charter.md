# Kayak Python Charter

This note defines what `kayak` for Python should be, how it should be used,
and how it should relate to the rest of the Kayak codebase.

It is a product-positioning note, not a packaging trace.
The goal is to keep the public story stable even while the internal engine
keeps evolving.

For the implementation sequence behind this position, see
[docs/python_sdk_roadmap.md](python_sdk_roadmap.md).

## Core Claim

`kayak` for Python should make late interaction feel native in Python.

That means:
- developers can `pip install kayak` and immediately work with late-interaction
  objects in normal Python code
- the API should feel natural to NumPy and PyTorch users
- the API should still preserve the structure that makes late interaction
  different from dense tensor math

Reason:
- the current code already exposes first-class `LateQuery`, `LateDocuments`,
  `LateIndex`, and `LateScores` objects
- the repo already keeps layouts and backends explicit
- flattening everything into a fake tensor abstraction would erase the repo's
  main technical advantage rather than clarify it

## Mission

The mission of Kayak Python is:

Provide the canonical open Python programming model for late-interaction
retrieval, with explicit late-interaction objects, exact reference semantics,
and optional Mojo acceleration.

In practical terms, that means `kayak` should be the package people use when
they want to:
- build late-interaction queries and document sets
- pack indexes
- build explicit local candidate generators and search plans
- run exact MaxSim scoring and search
- compare layouts and backends
- write retrieval code in Python without needing to understand the full engine
  internals

Reason:
- this is the part of the repo that answers the "I want to code with it"
  demand directly
- it gives the project a public developer surface that is useful even before a
  full hosted product exists

## Philosophy

### 1. Late interaction is the primitive

Kayak Python should not pretend that late interaction is just dense tensor
arithmetic with different branding.

The public model should stay centered on:
- ragged query vector counts
- ragged document vector counts
- explicit layouts
- MaxSim semantics

In other words, the primitive is token-level MaxSim over explicit query and
document groups, not a generic rerank hook with hidden structure.

Reason:
- those are the invariants that control retrieval quality, index size, memory
  cost, and latency
- hiding them would make the library easier to market but harder to reason
  about correctly

### 2. Python ergonomics, Mojo kernels

Users should write normal Python and choose between explicit backends.

The intended experience is:
- Python objects and operations at the top
- NumPy reference behavior always available
- Mojo acceleration when the environment supports it

Reason:
- that gives the package a broad usable baseline
- it also keeps the fast path real instead of hypothetical
- it avoids forcing every user to adopt Mojo before they can even try Kayak

### 3. Explicit over magic

Kayak Python should prefer explicit layout and backend choices over hidden
dispatch.

That means avoiding:
- hidden exact vs approximate switching
- implicit layout conversion
- overloaded tensor algebra that hides MaxSim semantics
- backend auto-selection that changes behavior silently

Reason:
- the repo is optimized for profiling, validation, and systems clarity
- hidden dispatch makes benchmarking and debugging harder

### 4. Small stable surface, rich internal engine

The public Python surface should remain narrow even if the monorepo grows.

The stable public entrypoint is:
- `import kayak`

The unstable internal layers remain:
- `kayak_bridge`
- the top-level Mojo engine package
- service and storage internals

Reason:
- a narrow public surface is easier to document, test, and version
- it prevents the open SDK from accidentally freezing internal engine details

## Product Split

Kayak should be described as one platform with two clearly different product
surfaces.

### 1. Kayak Python

This is the open, developer-facing SDK.

It should own:
- explicit late-interaction objects
- local exact scoring and search
- explicit local candidate generation and search plans
- layout conversion
- NumPy and PyTorch input ergonomics
- optional Mojo-backed acceleration
- small reproducible evaluation and validation helpers

It should not own:
- multi-tenant serving concerns
- collection lifecycle orchestration
- snapshots, compaction, and service operations as the main user story

Reason:
- the current Python API already models local late-interaction work, not hosted
  collection administration
- keeping the SDK focused makes it useful to researchers and application
  developers immediately

### 2. Kayak Engine

This is the hosted and operational retrieval system.

It should own:
- collections
- storage
- snapshots
- planning
- serving
- deployment and scaling concerns

It may remain proprietary even if Kayak Python is public.

Reason:
- the current repo already contains explicit engine-side service and collection
  contracts
- those concerns have a different cadence, stability boundary, and commercial
  value than the Python SDK

## Naming And Story

The package name `kayak` should mean the Python late-interaction SDK first.

The simplest public story is:
- `kayak` is the Python SDK for late-interaction retrieval
- Kayak Engine is the hosted or operational system that can exist behind the
  scenes or behind a service boundary

That is better than making `kayak` read primarily like a thin API client.

Reason:
- the current demand is about making late interaction programmable and usable
  in code
- the current implementation already matches a local SDK better than a hosted
  API client
- a thin-client-first story would understate the strongest part of the repo

## How Kayak Python Should Be Used

The default intended usage is local and explicit:

```python
import kayak

query = kayak.query(query_vectors)
documents = kayak.documents(doc_ids, document_vectors)
index = documents.pack()

scores = kayak.maxsim(query, index, backend=kayak.NUMPY_REFERENCE_BACKEND)
hits = kayak.search(query, index, k=10)
```

For `dim128`-optimized layouts:

```python
flat_query = query.to_layout("flat_dim128")
hybrid_index = index.to_layout("hybrid_flat_dim128")

scores = kayak.maxsim(
    flat_query,
    hybrid_index,
    backend=kayak.MOJO_EXACT_CPU_BACKEND,
)
```

The primary use cases are:
- research code that wants late interaction as a first-class abstraction
- offline evaluation and explicit candidate-window rescoring pipelines
- exact local validation against a reference backend
- application code that wants retrieval semantics without directly depending on
  the whole hosted engine

## Non-Goals

Kayak Python should not become:
- a generic tensor framework
- a vector database with a late-interaction veneer
- a grab-bag client for every internal service concept
- a package that requires Mojo just to run the reference path
- an API that hides vector count, layout, or backend choices

Reason:
- each of those directions would make the library broader, but also blur the
  one thing it should be uniquely good at

## Open Source Boundary

If the project chooses an open-source/public split, the safest boundary is:

Public:
- `python/kayak/`
- the tested public API exposed through `import kayak`
- the NumPy reference backend
- optional Mojo-backed exact local acceleration where packaging permits
- public docs, examples, and validation harnesses for the SDK

Potentially private or slower-moving:
- hosted engine internals
- service implementations
- deployment infrastructure
- proprietary storage and serving features

Reason:
- this follows the current code split more naturally than trying to publish the
  entire monorepo as one indiscriminate surface
- it also lets the Python package gain adoption without forcing a premature
  open/private decision on every engine module

## Roadmap Logic

The near-term plan for Kayak Python should be:

1. Stabilize the current object model.
   Reason: `LateQuery`, `LateDocuments`, `LateIndex`, and `LateScores` are the
   real public contract.

2. Keep the NumPy path as the always-works reference.
   Reason: the public package needs one backend with minimal operational
   assumptions.

3. Keep Mojo as explicit acceleration, not as the only way to use the package.
   Reason: performance is important, but adoption dies if the first-run path is
   fragile.

4. Add only a small number of core operations.
   Suggested center: `pack`, `to_layout`, `select`, `maxsim`, `search`.
   Reason: a small API is easier to make coherent and stable.

5. Add a remote client layer only after the hosted service contract stabilizes.
   Reason: the current SDK story is stronger than the current public-service
   story, so the client should be additive rather than identity-defining.

## Decision Rule

When deciding whether a feature belongs in Kayak Python, ask:

Does this make late interaction easier to program against in Python without
hiding the important retrieval structure?

If yes, it probably belongs in `kayak`.
If it mainly serves hosting, deployment, or internal engine operations, it
probably belongs in the engine layer instead.
