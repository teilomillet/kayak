# Filter Allowlist Indices For Centroid Fallback

## Claim

Keeping sorted `matching_doc_indices` in the document-filter allowlist lets the
centroid inactive-doc fallback avoid scanning every document when a selective
filter is active.

## Why this change

- The filter runtime in `kayak/collections/document_filter_runtime.mojo` already
  computed sorted matching doc indices and then discarded them.
- The centroid stage-1 fallback in
  `kayak/planning/execution_centroid_family.mojo` still scanned
  `range(len(doc_ids))` even when only a small filtered subset was allowed.
- Threading the existing sorted indices through the allowlist contract is a
  narrower and safer change than inventing a new filter-side artifact or making
  stage-1 infer sparsity indirectly from flags.

## Implementation

- `DocumentFilterAllowlist` now stores `matching_doc_indices` and validates that
  they are strictly ascending, in bounds, and aligned with allowed flags.
- `candidate_generation_for_centroid_family(...)` now keeps both
  `allowed_flags` and `allowed_doc_indices`.
- The centroid inactive-doc fallback uses `allowed_doc_indices` when a filter is
  active, while preserving the old full scan for the match-all path.
- For small active sets, the centroid inactive-doc fallback now sorts active doc
  indices once and merges them against the sorted allowlist instead of building
  a dense active bitmap. The merge path is intentionally bounded by
  `MAX_SORTED_ACTIVE_DOC_INDICES_FOR_MERGE = 128` so larger active sets keep the
  previous bitmap path.

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_document_filter_index.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
pixi run mojo -I . benchmarks/profile_centroid_filtered_fallback.mojo
```

Observed results:

- `tests/test_document_filter_index.mojo`: `5/5` passed
- `tests/test_collection_search_plan.mojo`: `34/34` passed
- `tests/test_service_runtime.mojo`: `25/25` passed

Benchmark fixture:

- `document_count = 50_000`
- `filter_match_count = 64`
- `active_match_count = 8`
- `candidate_k = 40`

Measured benchmark output before the merge refinement:

- `dense_filtered_fallback`: `4.711500332358891e-05 s`
- `indexed_filtered_fallback`: `2.2025943797296148e-05 s`
- speedup: `2.1390685346873823x`

Measured benchmark output after the merge refinement:

- `dense_filtered_fallback`: `6.13251813121943e-05 s`
- `indexed_filtered_fallback`: `2.2564732049023392e-05 s`
- `indexed_merge_filtered_fallback`: `1.2721434753201794e-06 s`
- indexed speedup over dense: `2.7177447167979323x`
- indexed-merge speedup over dense: `48.20618310899223x`

## Interpretation

This is a validated microbenchmark win on the exact fallback path the code
changed. It is not an end-to-end search claim by itself, but it shows that the
new allowlist contract and bounded merge path remove real filtered fallback
work rather than only making the code cleaner. The large indexed-merge speedup
is specific to the selective-filter, small-active-set regime that the merge
threshold targets.
