# Training-Free Sequence Compression Primitives

## Objective

Align the current implementation with the first paper target documented in:

- `docs/recent_paper_targets.md`

That note chose the training-free sequence compression family as the first
paper-shaped implementation target, specifically because it should fit the
document-representation transform seam cleanly.

The problem before this step was:

- the benchmark surface already existed
- but the method surface itself still lived mostly inside one benchmark module

That was weaker than the latent-proxy stage-1 work, where the primitive
boundary was extracted explicitly before further optimization.

## Why This Extraction Was Justified

This change follows the wall described in:

- `docs/architecture/extensibility_wall.md`

That architecture note says token pooling, pruning, and similar compression
methods should land at the **document-representation transform boundary**.

So the next sound step was not:

- add another compression benchmark variant

It was:

- extract an explicit transform-boundary primitive layer for the
  training-free sequence compression family

## What Changed

Added a new transform-boundary primitive module:

- `kayak/collections/training_free_sequence_compression.mojo`

New primitive surface:

- `TrainingFreeSequenceCompressionSpec`
- `training_free_sequence_compression_spec(...)`
- `training_free_sequence_compression_transforms(...)`
- `apply_training_free_sequence_compression_to_documents(...)`
- `apply_training_free_sequence_compression_to_packed_index(...)`
- `pool_factor_for_target_document_vector_budget(...)`
- `supported_training_free_sequence_compression_token_pooling_policies()`
- `default_training_free_sequence_compression_policies()`

New method constants now owned by the collections seam:

- `TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT`
- `TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING`
- `TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING`

Design choice:

- the spec carries `available_document_vector_count` explicitly

Reason:

- this repo treats vector count as a first-class design axis
- the benchmark note in `docs/recent_paper_targets.md` explicitly frames the
  paper surface around requested and realized vector budgets

## Benchmark Rewiring

Rewired:

- `kayak/benchmarks/training_free_sequence_compression_json.mojo`
- `benchmarks/browsecomp_plus_gold_training_free_sequence_compression.mojo`

What changed in practice:

- the benchmark summary builder now creates a
  `TrainingFreeSequenceCompressionSpec`
- the benchmark lowers that spec into ordinary
  `DocumentRepresentationTransformManifest` values
- the transformed packed index is built through
  `apply_training_free_sequence_compression_to_packed_index(...)`

Why this is better:

- the paper-method definition is now owned by the transform seam
- the benchmark file is now responsible mainly for measurement and reporting
- future benchmarks or runtime experiments can reuse the same compression spec
  without copying method logic

## Verification

### Focused primitive tests

Added:

- `tests/test_training_free_sequence_compression_primitives.mojo`

Ran:

```bash
pixi run mojo -I . tests/test_training_free_sequence_compression_primitives.mojo
```

Passed:

- `3` tests run
- `3` passed

What they verify:

- full-exact lowers to no transforms and normalizes to the available budget
- token-pooling specs derive the expected pool factor and manifest fields
- applying the spec matches direct transform-runtime execution

### Existing regression tests

Ran:

```bash
pixi run mojo -I . tests/test_training_free_sequence_compression_json.mojo
pixi run mojo -I . tests/test_document_representation_transform.mojo
```

Passed:

- `3/3` in `test_training_free_sequence_compression_json.mojo`
- `3/3` in `test_document_representation_transform.mojo`

Reason this matters:

- the existing paper-shaped JSON summary contract still works
- the lower transform seam still accepts the extracted method definitions

## Real Benchmark Smoke Check

Ran:

```bash
pixi run mojo -I . benchmarks/browsecomp_plus_gold_training_free_sequence_compression.mojo
```

Important note:

- this was a **single raw smoke run**, not a quiet-wrapper timing decision
- it verifies benchmark continuity and artifact production, not final
  comparative latency conclusions

Observed outcome:

- the benchmark completed successfully
- it wrote:
  `.cache/kayak/browsecomp_plus_gold_training_free_sequence_compression.json`

Structural facts checked from the produced artifact:

- `summary_count = 22`
- methods present:
  - `full_exact`
  - `prefix_pruning`
  - `token_pooling`
- policies present:
  - `""`
  - `prefix`
  - `hierarchical`
  - `sequential`
- minimum requested budget: `4`
- maximum requested budget: `175`

This confirms that the real paper-shaped surface still covers:

- the baseline
- pruning
- both supported pooling variants
- the full budget ladder

## Decision

Keep the new primitive layer.

Reason:

- it matches the transform-boundary seam chosen in the architecture docs
- it aligns the implementation with `docs/recent_paper_targets.md`
- it removes benchmark-only ownership of the paper-method surface
- it preserves the existing benchmark contract and artifact generation

## Conclusion

This step does not claim a new compression algorithm or a new benchmark win.

The narrower verified conclusion is:

- the first recent-paper target is now represented as an explicit primitive
  layer at the document-representation transform boundary
- the benchmark surface is now downstream of that primitive layer instead of
  defining the method family itself

That is a better foundation for the next paper-shaped compression step, because
new training-free methods can now plug into the same explicit spec and
lowering path rather than extending one benchmark file ad hoc.
