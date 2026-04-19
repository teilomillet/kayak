# Multi-Vector Index Compression Primitives

## Objective

Check the second compression-family target in:

- `docs/recent_paper_targets.md`

The question was not "can we add another benchmark variant?".

The question was:

- does the current primitive still hold when we plug the
  `Multi-Vector Index Compression in Any Modality` paper family into it?

That is the right falsification test because the first implemented family was
still entirely training-free and mostly transform-local.

## Sources Checked

Primary paper source checked on `2026-04-19`:

- arXiv abstract page for `2602.21202`

Public reference implementation sources checked on `2026-04-19`:

- `README.md`
- `src/arguments.py`
- `src/encoder/base_encoder.py`
- `src/encoder/resize_encoder.py`
- `src/encoder/select_encoder.py`

Why these sources were enough for this step:

- the immediate question was seam classification, not model-quality
  reproduction
- the abstract establishes the method family
- the public code establishes which methods are post-encoding transforms versus
  encoder-bound learned behaviors

## Verified Family Split

Verified from the paper abstract:

- the family includes:
  - sequence resizing
  - memory tokens
  - hierarchical pooling
  - attention-guided clustering

Verified from the public reference code:

- hierarchical pooling is implemented as a pooling operation over already
  produced token embeddings
- sequence resizing is implemented as a learned encoder module that projects the
  sequence dimension with MLP or linear weights
- memory tokens depend on appended tokens produced by the encoder
- attention-guided clustering depends on Universal Query tokens and output-layer
  attention

Inference from those verified facts:

- the paper family is **not** purely one post-encoding transform family in the
  way the first local reading suggested
- in `kayak` terms, the family spans at least two boundaries:
  - stored-document-representation transforms
  - encoder-bound learned compression

## Decision

Do not pretend all four methods can already lower through
`DocumentRepresentationTransformManifest`.

Instead:

- add a paper-shaped family module that classifies all methods explicitly
- allow only the stored-representation-executable subset to lower into ordinary
  transforms
- force explicit failure for encoder-bound methods until the repo has a sound
  learned compression path

That is a stronger primitive test than adding another transform alias, because
it checks whether the seam can represent negative cases cleanly.

## What Changed

Added:

- `kayak/collections/multi_vector_index_compression.mojo`
- `kayak/collections/document_encoder_compression.mojo`

New paper-family surface:

- `MultiVectorIndexCompressionSpec`
- `MultiVectorIndexCompressionLowering`
- `supported_multi_vector_index_compression_methods()`
- `stored_representation_multi_vector_index_compression_methods()`
- `multi_vector_index_compression_method_execution_boundary(...)`
- `multi_vector_index_compression_method_is_stored_representation_executable(...)`
- `require_multi_vector_index_compression_method_stored_representation_executable(...)`
- `multi_vector_index_compression_lowering(...)`
- `multi_vector_index_compression_spec(...)`
- `multi_vector_index_compression_transforms(...)`
- `apply_multi_vector_index_compression_to_documents(...)`
- `apply_multi_vector_index_compression_to_packed_index(...)`

Method coverage is now explicit:

- `full_exact`
- `hierarchical_pooling`
- `sequence_resizing`
- `memory_tokens`
- `attention_guided_clustering`

Boundary classification is now explicit:

- `stored_representation`
- `encoder`

Collection and segment provenance is now explicit too:

- `CollectionManifest` carries `document_encoder_compression`
- `SealedSegmentManifest` carries `document_encoder_compression`
- encoder-bound paper methods now lower into that manifest instead of being
  misrepresented as plain document transforms
- publish, compaction, and snapshot-bundle import/export paths now preserve
  that manifest instead of silently falling back to `none`
- the paper family now exposes one explicit lowering result instead of forcing
  callers to stitch transform lowering and encoder-compression lowering
  manually
- Mojo-side collection mirrors, service collection creation, and lifecycle
  reporting now preserve and surface the same provenance
- the Python hosted-engine loader and HTTP collection-creation payload now
  accept the same `document_encoder_compression` manifest shape and thread it
  into the Mojo service boundary instead of truncating it to defaults

## Why This Primitive Is Better

Before this change:

- the first paper-family layer could encourage the false idea that "compression
  family" means "runtime transform family"

After this change:

- a paper family can be represented even when only part of it is executable on
  stored vectors
- benchmark code can measure only the executable subset without silently
  inventing support for the rest
- the missing primitive is now precise:
  encoder-bound learned compression needs a separate seam from plain document
  transforms
- persisted hosted collection metadata can record encoder-bound document
  compression provenance explicitly instead of hiding it inside `model_name`
- one Mojo primitive now owns the full document-side lowering split for the
  family, which reduces the risk that future callers combine both seams
  incorrectly
- the hosted Mojo service contract can now create and observe collections with
  encoder-bound document compression provenance instead of collapsing that
  state during creation or retention updates

## Benchmark Surface

Added:

- `kayak/benchmarks/multi_vector_index_compression_json.mojo`
- `benchmarks/browsecomp_plus_gold_multi_vector_index_compression.mojo`

Important scope choice:

- the benchmark surface only executes methods classified as
  `stored_representation`

Reason:

- benchmarking encoder-bound methods through the stored-document runtime would
  be a false claim, not an approximation

## Verification

Added focused primitive tests:

- `tests/test_multi_vector_index_compression_primitives.mojo`

Added benchmark-surface tests:

- `tests/test_multi_vector_index_compression_json.mojo`

Added Python transport checks:

- `python/tests/test_prepared_exact_search_session.py`
- `python/tests/test_hosted_engine_http.py`

These checks were intended to prove:

- all paper methods are classified
- hierarchical pooling lowers to the current document-transform runtime
- encoder-bound methods fail explicitly when forced through the wrong seam
- the benchmark JSON surface reports the boundary and still preserves the
  vector-budget accounting contract
- the hosted Python binding and HTTP transport can create collections whose
  persisted lifecycle state still reports encoder-bound compression provenance

## Conclusion

This step does **not** claim a faithful reproduction of the full paper.

The verified narrower conclusion is:

- the second compression-family target is now represented in `kayak`
- checking all paper cases shows that the right primitive is **not**
  "one universal transform-only compression spec"
- the stronger primitive is:
  - a paper-family layer
  - explicit boundary classification
  - transform lowering only for the stored-representation-executable subset

That is a more epistemically sound base for future AGC, memory-token, or
sequence-resizing work than treating them as if they were already ordinary
document transforms.
