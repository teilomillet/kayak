# Imputed Flat Overflow Heap Selector

## Claim

`centroid_postings_imputed_flat` now uses an overflow-only shortlist kernel that
keeps the top `bound` centroids in a worst-first heap while scanning and sorts
that bounded set once at the end.

The claim is intentionally narrow:

- this improves the flat imputed selector when `centroid_count > bound`
- it preserves the current tie semantics exactly
- it does **not** claim a broad public-frontier win on the current public slices,
  because those slices sit at or below the current bound and do not materially
  exercise the overflow path

I explicitly did **not** keep the same kernel on the nested imputed selector.
The wrapped evidence did not justify broadening the change beyond the promoted
flat path.

## Why This Scope

I first tried the overflow shortlist kernel on both nested and flat imputed
selectors. That version passed correctness, and it clearly helped on a
high-centroid synthetic probe, but the wrapped public primitive table was too
ambiguous to justify publishing the nested change.

Direct code and artifact inspection showed why the public evidence was weak:

- `SciFact` primitive snapshot centroid count: `127`
- `FIQA` primitive snapshot centroid count: `128`
- `LIMIT-small` primitive snapshot centroid count: `111`
- BrowseComp-Plus Gold search-breakdown snapshot centroid count: `128`

Those counts come from the local mirrored manifests:

- `.cache/kayak/scifact_real_subset_primitives/.../centroid_postings/manifest.tsv`
- `.cache/kayak/fiqa_real_subset_primitives/.../centroid_postings/manifest.tsv`
- `.cache/kayak/limit_small_real_subset_primitives/.../centroid_postings/manifest.tsv`
- `.cache/kayak/browsecomp_plus_gold_workspace_search_breakdown/.../centroid_postings/manifest.tsv`

With `bound = min(centroid_count, 128)`, those slices are at or below the bound.
So the honest design is:

- keep the old sorted insertion path for non-overflow cases
- use the heap-based overflow kernel only on the flat selector when
  `centroid_count > bound`

That keeps the optimized code aligned with the regime where the work reduction
is real.

## Implementation

Files:

- `kayak/planning/imputed_centroid_shortlist.mojo`
- `kayak/planning/centroid_postings_imputed_flat_stage.mojo`
- `tests/test_imputed_high_centroid_probe.mojo`

What changed:

- added a local worst-first centroid shortlist helper with explicit tie rules
  - lower score is worse
  - on equal scores, the larger centroid index is worse
- gated the flat imputed selector:
  - if `centroid_count <= bound`, keep the old sorted insertion path
  - if `centroid_count > bound`, use the heap maintenance kernel and sort the
    bounded shortlist once before finalization
- added tie-heavy regression tests on an explicit `129`-centroid overflow case
  for both:
  - keeping earlier equal-score centroids
  - evicting the current tail when a later better centroid arrives

Why those tie rules matter:

- the pre-existing sorted insertion path keeps earlier equal-score centroids
- the overflow kernel must evict the same element the old tail would have
  evicted, or the selector contract changes

## Validation

Commands run:

```bash
pixi run mojo -I . tests/test_imputed_high_centroid_probe.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_service_runtime.mojo
```

Observed results:

- `tests/test_imputed_high_centroid_probe.mojo`: `8/8` passed
- `tests/test_collection_search_plan.mojo`: `38/38` passed
- `tests/test_service_runtime.mojo`: `26/26` passed

## Wrapped Overflow Evidence

This is the benchmark that directly exercises the target regime.

Sources:

- baseline worktree on `f9f756d`:
  `/tmp/kayak-overflow-baseline/.cache/kayak/bench_quiet/20260415T014052Z/run_1.txt`
- current worktree:
  `.cache/kayak/bench_quiet/20260415T020421Z/run_1.txt`

Wrapped `profile_imputed_selection_high_centroid.mojo`:

- overflow probe shape:
  - `slice_name = imputed_probe_centroids180_docs544`
  - `vector_dim = 180`
  - `centroid_count = 180`

Measured results:

- nested selector sanity check:
  - before: `9.795119813387997e-05 s`
  - after: `9.763706055665207e-05 s`
  - delta: `-0.314 us` (`-0.32%`)
- flat selector target:
  - before: `1.8713588467916825e-04 s`
  - after: `1.7160220760920622e-04 s`
  - delta: `-15.534 us` (`-8.30%`)

What this verifies:

- the flat overflow kernel is materially faster on the regime it targets
- the nested selector is effectively unchanged after narrowing scope back to the
  old path

## Public-Slice Interpretation

I also reran the wrapped public primitive benchmark:

- current source: `.cache/kayak/bench_quiet/20260415T014544Z/run_1.txt`
- previous source: `.cache/kayak/bench_quiet/20260414T235004Z/run_1.txt`

I am **not** using that table as the performance claim for this slice.

Reason:

- those public slices are `111–128` centroids wide
- the overflow-only kernel is gated off there by design
- the resulting wrapped differences are small and host-noise-sensitive, so they
  are not decision-quality evidence for or against the overflow kernel itself

The right interpretation is:

- current public slices remain effectively on the previous small-count path
- the new value is shape-specialized behavior when centroid count exceeds the
  current imputed bound

## Conclusion

This slice is worth keeping as a narrow kernel improvement.

The evidence supports:

- specializing the promoted flat imputed selector by shape
- keeping the policy and stage contracts unchanged
- refusing to generalize the claim beyond the measured overflow regime

It does **not** support claiming a broad public benchmark win today, and this
trace intentionally does not make that claim.
