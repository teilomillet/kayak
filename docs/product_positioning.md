# Product Positioning

Status: working product note  
Date: `2026-04-13`

This note defines the product category that `kayak` should occupy and the
deployment trust model it should optimize for.

It is intentionally epistemic:
- category claims are separated from verified repository facts
- competitor comparisons are used to clarify the category, not to imply
  benchmark superiority that the repo has not yet measured
- privacy language stays precise about what vectors do and do not hide

Related repo notes:
- [docs/python_sdk_charter.md](python_sdk_charter.md)
- [docs/architecture/service_api.md](architecture/service_api.md)
- [docs/architecture/search_plan_semantics.md](architecture/search_plan_semantics.md)
- [docs/hosted_engine_grand_plan.md](hosted_engine_grand_plan.md)

## Sources Checked

Checked on `2026-04-13`:

- Chroma official docs:
  - https://docs.trychroma.com/cloud/getting-started
  - https://docs.trychroma.com/guides/deploy/docker
  - https://docs.trychroma.com/production/chroma-server/client-server-mode
  - https://docs.trychroma.com/cloud/sync/s3
- Qdrant official docs:
  - https://qdrant.tech/documentation/cloud/
  - https://qdrant.tech/documentation/hybrid-cloud/
  - https://qdrant.tech/documentation/concepts/vectors/
- Firnflow official repository:
  - https://github.com/gordonmurray/firnflow

Reason:
- these products cover the closest adjacent categories that matter for Kayak's
  outward story:
  - easy developer database or cloud search product
  - serious deployable vector infrastructure
  - object-storage-native search service

## Executive Summary

`kayak` should be positioned as:

**a late-interaction-native retrieval engine with a clean Python SDK and a
deployable search service**

That is more precise than:
- "vector database"
- "RAG ingestion platform"
- "model repository"
- "just an API client"

Reason:
- the Python surface is already becoming a programming model for explicit
  late-interaction objects
- the engine surface already has collection, snapshot, planning, and service
  contracts
- the most important differentiator is not generic vector storage, but that
  the whole system is shaped around late interaction itself

## What Companies Actually Buy

Companies do not buy retrieval infra only because it has a strong kernel.

They buy confidence in:
- deployment options
- data boundary clarity
- correctness and explainability
- operational recovery
- predictable latency and cost

Decision:
- "peace of mind" should be treated as a first-class product requirement, not
  as a marketing afterthought

For Kayak, peace of mind means:
- one explicit exact reference path
- approximate paths that are labeled, measurable, and explainable
- explicit collection and tenant boundaries
- snapshots, compaction, and recovery surfaces
- health, metrics, and operational reporting
- deployment modes that are easy to understand

## Category Definition

Kayak should be one platform with two product surfaces.

### 1. Kayak Python

This is the open SDK.

It should be the thing people install when they want to:
- build `LateQuery`, `LateDocuments`, and `LateIndex` objects
- validate exact MaxSim locally
- compare layouts and backends
- prepare upload or serving artifacts
- write late-interaction retrieval code directly in Python

This is already aligned with:
- [docs/python_sdk_charter.md](python_sdk_charter.md)

### 2. Kayak Engine

This is the deployable retrieval service.

It should be the thing companies use when they want to:
- host multi-vector search artifacts
- run explicit late-interaction search plans
- get exact or approximate results through a stable API
- isolate tenants, snapshots, and operational state

This is already aligned with:
- [docs/architecture/service_api.md](architecture/service_api.md)
- [docs/architecture/search_plan_semantics.md](architecture/search_plan_semantics.md)

## What Kayak Is Not

Kayak should **not** be positioned primarily as:

- a generic vector database with late interaction as one feature
- a generic document parsing and ingestion SaaS
- a thin wrapper over an external search API
- a training-first model framework

Reason:
- those categories already have strong incumbents
- they also hide the main thing Kayak can make explicit:
  late-interaction-native semantics across local code, indexing, planning, and
  serving

## Comparison To Adjacent Products

The goal of these comparisons is category clarity, not leaderboard claims.

### Chroma

What Chroma appears to optimize for:
- low-friction developer adoption
- self-hosted and managed-cloud deployment
- integrated sync and ingestion workflows
- broad search ergonomics for common application teams

Why it matters:
- Chroma is a good benchmark for product simplicity and deployability
- it answers the buyer question "can I use this without worrying too much?"

What Kayak should learn from Chroma:
- short path from local development to cloud use
- simple deployment guides
- clear operational limits and defaults
- reduction of adoption anxiety

What Kayak should not copy from Chroma:
- a broad "just store data and search it" story that blurs late-interaction
  semantics into a generic retrieval layer

Inference:
- Chroma is primarily a database or cloud retrieval product
- Kayak should instead be a late-interaction-native engine with stronger
  semantic and systems explicitness

### Qdrant

What Qdrant appears to optimize for:
- production-grade vector infrastructure
- managed cloud and hybrid-cloud deployment flexibility
- strong operational guarantees and cloud controls
- broad vector-database use cases, including multivectors

Verified from official docs:
- Qdrant has managed cloud and hybrid-cloud deployment surfaces
- Qdrant supports multivectors with a `max_sim` comparator

Why it matters:
- Qdrant is close to the category customers will use to understand Kayak
- it already shows that buyers care about privacy, sovereignty, control,
  backups, scaling, and disaster recovery

What Kayak should learn from Qdrant:
- deployment choice matters
- privacy and data sovereignty must be explicit
- operational features are part of the product, not only the implementation

What Kayak should not copy from Qdrant:
- the category framing of "general vector database first"

Inference:
- Qdrant is a strong reference for deployment trust
- Kayak should differentiate by centering late interaction as the native
  abstraction rather than as one supported vector mode

### Firnflow

What Firnflow appears to optimize for:
- object-storage-backed search economics
- RAM to NVMe to object-store tiering
- namespace isolation
- API-first search service deployment

Verified from the repository README:
- Firnflow presents itself as a multi-tenant vector and full-text search engine
  backed by object storage
- its service surface includes upsert, query, indexing, compaction, metrics,
  and warmup

Why it matters:
- Firnflow is structurally close to the kind of hosted service shape Kayak may
  want
- it is a useful reference for service topology, cache economics, and
  namespace design

What Kayak should learn from Firnflow:
- object-store economics can be a product story
- clear namespace and cache semantics help the deployment story
- API-first search infrastructure is legible to infra buyers

What Kayak should not copy from Firnflow:
- a generic vector or full-text engine story that leaves late interaction as
  just another workload

Inference:
- Firnflow is the closest comparison for service topology
- Qdrant is the closest comparison for production infrastructure expectations
- Chroma is the closest comparison for ease of adoption

## Recommended Data Boundary

The cleanest product boundary for Kayak Engine is:

- customers may keep raw documents on their side
- customers may encode documents on their side
- Kayak may ingest:
  - vectors or packed multi-vector artifacts
  - ids
  - optional lightweight metadata
  - optional text sidecars only when a richer verifier path is explicitly
    enabled
- Kayak search returns:
  - ids
  - scores
  - optional evidence or explain data

Reason:
- this keeps the default trust boundary narrow
- it avoids turning Kayak into a mandatory raw-data host before that is
  necessary
- it fits the repo's current exact-first late-interaction focus

Important caution:
- vectors are still sensitive data
- "Kayak does not store raw documents by default" is defensible
- "Kayak never has access to user data" is not precise if Kayak stores vectors
  or metadata

## Deployment Promise

If Kayak wants to be trusted by companies, the deployment promise should be:

1. Clear modes
   - local development
   - self-hosted production
   - later, managed or private deployment

2. Stable retrieval semantics
   - exact reference path
   - approximate paths explicitly labeled
   - explain and debug surfaces for search behavior

3. Operational recovery
   - snapshots
   - health
   - metrics
   - compaction and retention surfaces
   - upgrade and rollback story

4. Explicit data ownership
   - who encodes
   - what Kayak stores
   - whether raw text sidecars are enabled
   - what tenant boundary is enforced

This is the "peace of mind" bar.

## Recommended Public Language

### Homepage one-liner

`kayak` is a late-interaction-native retrieval engine for teams that need
better search quality, explicit semantics, and a deployable service boundary.

### SDK framing

`pip install kayak` gives developers a Python-native programming model for
late interaction, with exact reference semantics and optional Mojo
acceleration.

### Engine framing

Kayak Engine hosts packed multi-vector search artifacts and serves explicit
late-interaction search over stable collection and snapshot boundaries.

### Short comparison framing

Kayak is to late interaction what a serious vector database is to embeddings,
except the system is built around multi-vector retrieval from the start.

## Near-Term Product Priorities

If product trust is the goal, the next high-leverage items are:

- document the default data boundary and trust model
- publish one clean self-hosted deployment path
- keep exact and approximate search semantics explicit in all public APIs
- keep health, metrics, snapshot, and recovery surfaces first-class
- make the public Python SDK and the engine deployment story readable without
  reading the whole repo

Reason:
- those items reduce adoption anxiety more directly than one more benchmark
  tweak or one more candidate-generator variant

## Final Position

The most coherent position today is:

- Chroma is a useful reference for ease of adoption
- Qdrant is a useful reference for production trust and deployment flexibility
- Firnflow is a useful reference for service topology and object-storage
  economics
- Kayak should compete by being the most explicit and deployable
  late-interaction-native engine, not by becoming a generic vector database
