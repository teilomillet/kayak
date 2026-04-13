# 2026-04-13: Public Benchmark Dataset Registry

## Why This Step Exists

Several benchmark entrypoints still encoded public-dataset facts directly:

- which dataset cache to load
- whether a text sidecar exists
- how to name the mirrored collection root
- whether a benchmark was reusing the correct text sidecar for that slice

That was workable while the surface was small, but it was not epistemically
sound enough for a modular late-interaction engine:

- capability and loaded state were conflated
- benchmark runners had to know dataset-specific branching details
- one verifier path reused the gold text corpus for the evidence slice

The goal of this step was to make the dataset seam explicit without baking in
more stage-1 architectural assumptions.

## What Changed

Added:

- [kayak/benchmarks/public_benchmark_dataset.mojo](../../kayak/benchmarks/public_benchmark_dataset.mojo)
- [tests/test_public_benchmark_dataset.mojo](../../tests/test_public_benchmark_dataset.mojo)

Updated:

- [kayak/benchmarks/__init__.mojo](../../kayak/benchmarks/__init__.mojo)
- [kayak/benchmarks/planner_benchmark_runner.mojo](../../kayak/benchmarks/planner_benchmark_runner.mojo)
- [benchmarks/browsecomp_plus_gold_planner_evidence.mojo](../../benchmarks/browsecomp_plus_gold_planner_evidence.mojo)
- [benchmarks/browsecomp_plus_gold_ceiling_comparison.mojo](../../benchmarks/browsecomp_plus_gold_ceiling_comparison.mojo)
- [benchmarks/browsecomp_plus_clause_text_verifier.mojo](../../benchmarks/browsecomp_plus_clause_text_verifier.mojo)
- [benchmarks/real_subset_search_json.mojo](../../benchmarks/real_subset_search_json.mojo)
- [benchmarks/real_subset_collection_storage_json.mojo](../../benchmarks/real_subset_collection_storage_json.mojo)

### A dataset registry is now first-class

`PublicBenchmarkDataset` now carries:

- dataset key
- collection-root stem
- explicit `has_text_sidecars`
- cache-load provenance
  - `loaded_task_from_storage`
  - `loaded_index_from_storage`
- stored task/index payloads
- optionally loaded text corpus

The loader API distinguishes:

- whether a dataset supports text sidecars at all
- whether the caller actually loaded a text corpus this time

That separation matters because clause-text benchmarks should be enabled only
when text is actually loaded, not merely because a dataset family happens to
support text.

### Text-sidecar loading now has guardrails

The registry now validates any loaded text corpus against the packed index:

- document count must match
- document order must match

This prevents a silent mismatch between text sidecars and vector indices from
passing through the benchmark surface as if it were valid evidence.

### Mirrored benchmark collections now share a common helper

`ensure_public_benchmark_dataset_collection_mirror(...)` centralizes:

- collection-root naming
- collection identity
- text-sidecar attachment
- optional frontier GEM sidecar construction

That lets benchmark entrypoints ask for a benchmark-ready mirror without each
script repeating the same collection plumbing.

### Planner and BrowseComp benchmark paths now use the registry

The planner benchmark runner no longer contains per-dataset cache/text branches.
It now:

- selects dataset keys from options
- loads datasets through the registry
- enables clause-text only when a text corpus is actually loaded

The BrowseComp gold planner-evidence and ceiling benchmarks also now use the
same dataset seam and the same mirror helper.

### The clause-text verifier no longer reuses the gold text sidecar for evidence

Previously, the verifier loaded the gold text corpus and used it for both:

- evidence slice reranking
- gold slice reranking

That was an unnecessary assumption.

It now loads and validates:

- the evidence text corpus for the evidence slice
- the gold text corpus for the gold slice

So the verifier is now aligned with the slice it is actually measuring.

### Architecture-neutral inventory benchmarks now use the registry too

The plain real-slice search summary and the collection-storage inventory now
load datasets through the registry rather than hardcoding five separate cache
calls.

This keeps the seam useful even for benchmark surfaces that are not making any
stage-1 choice.

## Validation

Unit tests:

```bash
pixi run mojo -I . tests/test_public_benchmark_dataset.mojo
pixi run mojo -I . tests/test_planner_benchmark_json.mojo
pixi run mojo -I . tests/test_planner_evidence_json.mojo
```

Real-path planner and verifier scripts:

```bash
pixi run bench_real_subset_planner_benchmark_smoke_raw
pixi run bench_browsecomp_plus_gold_planner_benchmark_raw
pixi run bench_browsecomp_plus_gold_planner_evidence_raw
pixi run mojo -I . benchmarks/browsecomp_plus_gold_ceiling_comparison.mojo
pixi run mojo -I . benchmarks/browsecomp_plus_clause_text_verifier.mojo
pixi run mojo -I . benchmarks/real_subset_search_json.mojo
pixi run mojo -I . benchmarks/real_subset_collection_storage_json.mojo
```

Observed results:

- all listed commands passed locally
- `.cache/kayak/public_planner_benchmark_smoke.json` contains `4` summaries, all
  with `stage2_kind = exact_late_interaction`
- `.cache/kayak/browsecomp_plus_gold_planner_benchmark.json` contains `40`
  summaries split across:
  - `20` `exact_late_interaction`
  - `20` `clause_text`
- `.cache/kayak/browsecomp_plus_gold_planner_evidence.json` contains the same
  `20/20` stage-2 split
- `benchmarks/browsecomp_plus_clause_text_verifier.mojo` completed with separate
  evidence and gold text corpora and reported:
  - evidence clause-text rerank `ndcg@10 = 0.2393`
  - gold clause-text rerank `ndcg@10 = 0.3638`
- `.cache/kayak/public_real_slice_benchmarks.json` contains `5` dataset
  summaries
- `.cache/kayak/public_real_slice_collection_storage.json` contains `5` dataset
  summaries

## Epistemic Boundary

This step proves:

- public benchmark dataset capabilities are now represented explicitly
- text-sidecar use is checked against the indexed document order
- planner and verifier scripts can share a generalized dataset seam
- architecture-neutral inventory benchmarks can also reuse that same seam

This step does **not** prove:

- that the current public benchmark registry is exhaustive
- that every remaining benchmark script should immediately be refactored onto it
- that the registry itself settles any stage-1 design question

It only establishes a better substrate for future refactors: dataset facts now
live behind an explicit, test-validated interface instead of being scattered
through individual benchmark entrypoints.
