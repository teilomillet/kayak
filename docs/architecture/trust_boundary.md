# Trust Boundary

Status: current product and service note  
Date: `2026-04-13`

This note defines the default trust boundary for Kayak.

It exists because "peace of mind" depends on what Kayak stores, what it does
not store, and what claims it can honestly make about customer data.

Related notes:
- [data_flow_io.md](data_flow_io.md)
- [service_api.md](service_api.md)
- [../product_positioning.md](../product_positioning.md)

## Core Rule

Kayak should default to:

- customer-managed raw documents
- customer-managed encoding
- Kayak-managed late-interaction search artifacts

Reason:
- this is the narrowest trustworthy boundary
- it matches the current engine shape, where the encoder boundary is external
- it avoids forcing Kayak to become a raw-document host before that is
  necessary

## What Kayak Stores By Default

Default hosted boundary:
- collection metadata
- encoded document vectors or packed multi-vector artifacts
- document ids
- snapshot and segment metadata

Optional by explicit choice:
- lightweight document metadata for filtering
- text sidecars for text-aware verification

Not required by default:
- full raw documents
- source files
- source connectors

## What Kayak Returns

Default return path:
- ids
- scores
- plan metadata

Optional richer return path:
- explain and debug data
- stage-level diagnostics
- verifier evidence coordinates

This means the normal product boundary can be:
- customer keeps the original content
- Kayak returns the best matching ids and scores

## What Kayak Should Not Claim

Kayak should **not** claim:
- "Kayak never sees user data"
- "vectors are not sensitive"
- "vector-only storage means zero privacy risk"

Reason:
- vectors can still leak information
- metadata can still be sensitive
- optional text sidecars clearly contain user content

Defensible language is:
- Kayak does not need to store raw documents by default
- Kayak can operate on encoded late-interaction representations
- vectors and metadata should still be treated as sensitive data

## Query Boundary

Current rule:
- the hosted search path accepts encoded queries
- the request now carries explicit `query_model_name`

Reason:
- the query encoder is external to the service
- the service should be able to reject a query that claims the wrong model
  space

Important consequence:
- if a caller wants Kayak not to see raw query text, they should avoid
  text-aware verifier paths or keep `query_text` empty

## Optional Text Sidecars

Text sidecars should remain opt-in.

Use them only when a richer path explicitly needs them, such as:
- clause-text verification
- future evidence-oriented reranking

Do not make text sidecars an implicit requirement for:
- exact MaxSim
- ordinary vector search
- vector-only hosted deployments

## Deployment Implications

The narrow default trust boundary implies three good deployment modes.

### 1. Local SDK only

The user keeps everything local.

### 2. Self-hosted engine

The user keeps:
- raw data
- encoded artifacts
- the running Kayak service

This is the strongest privacy and control path.

### 3. Hosted engine

Kayak may host:
- vectors
- ids
- optional metadata
- optional text sidecars

This is still a narrower boundary than a full raw-document ingestion product,
but it is not a zero-trust or zero-data boundary.

## Operator Requirements

Because vectors and metadata are still sensitive, a serious deployment should
eventually include:
- auth
- tenant isolation
- storage isolation
- explicit snapshot roots
- retention policy controls
- restore and deletion workflows

Those are product requirements, not only implementation details.

## Final Recommendation

Kayak should communicate the default trust boundary as:

- bring your own encoded representations
- keep raw documents on your side unless you explicitly opt into sidecars
- let Kayak handle late-interaction storage, search, and explain

That is the cleanest path to a trustworthy deployable product.
